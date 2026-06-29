// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/stage.h"
#include "nanousd/clips.h"
#include "nanousd/schema.h"
#include "nanousd/spline.h"
#include "nanousd/usd_parser.h"
#include "path_pool.h"   // for GetCachedPathElements (internal, no-copy element access)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace nanousd {

namespace {

std::vector<Path> NormalizePopulationMask(std::vector<Path> mask) {
    std::vector<Path> result;
    PathSet seen;
    for (Path path : mask) {
        if (path.IsEmpty()) continue;
        if (path.HasProperty()) path = path.GetPrimPath();
        if (!path.IsPrimPath()) continue;
        if (path.IsAbsoluteRoot()) {
            result.clear();
            result.push_back(Path::AbsoluteRoot());
            return result;
        }
        if (seen.count(path)) continue;
        seen.insert(path);
        result.push_back(path);
    }
    return result;
}

// Env-gated phase timer: prints to stderr only when NANOUSD_TIMING=1.
// No-op (zero overhead past one getenv) when disabled.
struct PhaseTimer {
    using clock = std::chrono::steady_clock;
    clock::time_point t0;
    bool enabled;
    PhaseTimer() : t0(clock::now()) {
        const char* v = std::getenv("NANOUSD_TIMING");
        enabled = (v && v[0] == '1');
    }
    void lap(const char* label) {
        if (!enabled) { t0 = clock::now(); return; }
        auto t1 = clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  [timing] %-30s %9.1f ms\n", label, ms);
        t0 = t1;
    }
};

// Per-run accumulators for diagnosing UsdPrim::GetAttributeNames cost.
// Reset when Traverse() starts; flushed if NANOUSD_TIMING=1 when the
// Stage is destroyed (see printer below).
struct AttrNamesStats {
    long long ns_entry_walk = 0;
    long long ns_schema = 0;
    long long ns_total = 0;
    long long entries_visited = 0;
    long long unique_names = 0;
    long long calls = 0;
    void reset() { *this = AttrNamesStats{}; }
};
static AttrNamesStats g_attrStats;

inline bool AttrTimingEnabled() {
    static bool on = [] {
        const char* v = std::getenv("NANOUSD_TIMING");
        return v && v[0] == '1';
    }();
    return on;
}

inline Path OpinionSourcePrimPath(const OpinionEntry& entry,
                                  const Path& primPath) {
    return entry.sourcePath.IsEmpty()
        ? entry.pathMapping->MapToSource(primPath)
        : entry.sourcePath;
}

inline Path OpinionSourcePath(const OpinionEntry& entry, const Path& path) {
    if (!path.IsPropertyPath()) return OpinionSourcePrimPath(entry, path);
    Path sourcePrimPath = OpinionSourcePrimPath(entry, path.GetPrimPath());
    if (sourcePrimPath.IsEmpty()) return Path();
    return sourcePrimPath.AppendProperty(path.GetPropertyName());
}

struct AttrNamesStatsPrinter {
    ~AttrNamesStatsPrinter() {
        if (!AttrTimingEnabled() || g_attrStats.calls == 0) return;
        auto ms = [](long long ns) { return ns / 1e6; };
        std::fprintf(stderr,
            "      [traverse] GetAttributeNames calls: %lld  (avg entries=%.1f unique=%.1f)\n",
            g_attrStats.calls,
            (double)g_attrStats.entries_visited / g_attrStats.calls,
            (double)g_attrStats.unique_names   / g_attrStats.calls);
        std::fprintf(stderr, "      [traverse]   entry walk:    %7.1f ms\n", ms(g_attrStats.ns_entry_walk));
        std::fprintf(stderr, "      [traverse]   schema:        %7.1f ms\n", ms(g_attrStats.ns_schema));
        std::fprintf(stderr, "      [traverse]   total:         %7.1f ms\n", ms(g_attrStats.ns_total));
    }
};
static AttrNamesStatsPrinter s_attrPrinter;

// Forward declarations of generic resolution helpers defined further
// down in their own anonymous namespace block. The early UsdProperty /
// UsdAttribute / UsdRelationship accessors call them, so they need a
// visible declaration here at the top of the file.
template <typename T>
static std::optional<ListOp<T>> ResolveListOpField(
        const CompositionGraph& graph,
        const PrimIndex* primIndex,
        const Path& path,
        const Token& fieldName,
        SpecType specType);

static bool HasAnyAuthoredField(
        const CompositionGraph& graph,
        const PrimIndex* primIndex,
        const Path& path,
        const Token& fieldName,
        SpecType specType);

std::vector<Token> GetAuthoredAppliedSchemasForPrim(
        const CompositionGraph* graph,
        const PrimIndex* primIndex,
        const Path& primPath,
        const Spec* fallbackSpec);

static Token ValueAsToken(const Value* value);

static std::optional<Token> GetStrongestAttributeTokenField(
        const CompositionGraph& graph,
        const PrimIndex* primIndex,
        const Path& attrPath,
        const Token& fieldName);

static Token ComputePrimColorSpaceName(const CompositionGraph* graph,
                                       const PrimIndex* primIndex,
                                       const Path& primPath);

static Role RoleForTypeName(std::string typeName) {
    if (typeName.size() >= 2 && typeName.substr(typeName.size() - 2) == "[]") {
        typeName.resize(typeName.size() - 2);
    }
    if (typeName == "color3d" || typeName == "color3f" ||
        typeName == "color3h" || typeName == "color4d" ||
        typeName == "color4f" || typeName == "color4h") {
        return Role::Color;
    }
    if (typeName == "normal3d" || typeName == "normal3f") {
        return Role::Normal;
    }
    if (typeName == "point3d" || typeName == "point3f") {
        return Role::Point;
    }
    if (typeName == "vector3d" || typeName == "vector3f") {
        return Role::Vector;
    }
    if (typeName == "frame4d") {
        return Role::Frame;
    }
    if (typeName == "texCoord2d" || typeName == "texCoord2f") {
        return Role::TexCoord;
    }
    return Role::None;
}

static void ApplyRoleFromTypeName(Value& value, const Token& typeName) {
    if (value.IsBlock()) return;
    Role role = RoleForTypeName(typeName.GetString());
    if (role != Role::None) value.SetRole(role);
}
} // namespace

const std::string Stage::s_empty;
const std::vector<Path> Stage::s_emptyPaths;

// ============================================================
// UsdProperty
// ============================================================

bool UsdProperty::IsCustom() const {
    return spec_ ? spec_->GetCustom() : false;
}

bool UsdProperty::IsHidden() const {
    return spec_ ? spec_->GetHidden() : false;
}

String UsdProperty::GetDisplayName() const {
    return spec_ ? spec_->GetDisplayName() : "";
}

String UsdProperty::GetDisplayGroup() const {
    if (!spec_) return "";
    return spec_->GetFields().GetOr<String>(FieldNames::displayGroup, "");
}

String UsdProperty::GetDocumentation() const {
    return spec_ ? spec_->GetDocumentation() : "";
}

// ============================================================
// UsdAttribute
// ============================================================

Token UsdAttribute::GetTypeName() const {
    if (spec_) return spec_->GetTypeName();
    if (hasSchemaDef_) return Token(schemaDef_.typeName);
    return Token("");
}

Variability UsdAttribute::GetVariability() const {
    if (spec_) return spec_->GetVariability();
    if (hasSchemaDef_) return schemaDef_.variability;
    return Variability::Varying;
}

const Value* UsdAttribute::GetDefault() const {
    // Walk the opinion stack for the first authored default.
    // Returns the raw value (including blocks) — block interpretation
    // is handled by Get() during full value resolution.
    if (graph_ && primIndex_) {
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePath(entry, path_);
            if (sourcePath.IsEmpty()) continue;
            const auto* spec = layer.GetAttributeSpec(sourcePath);
            if (!spec) continue;
            auto* authored = spec->GetField(FieldNames::defaultValue);
            if (authored) return authored;
        }
        return GetFallback();
    }

    // Legacy single-layer path
    if (spec_) {
        auto* authored = spec_->GetField(FieldNames::defaultValue);
        if (authored) return authored;
    }
    return GetFallback();
}

bool UsdAttribute::HasDefault() const {
    if (graph_ && primIndex_) {
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePath(entry, path_);
            if (sourcePath.IsEmpty()) continue;
            const auto* spec = layer.GetAttributeSpec(sourcePath);
            if (!spec) continue;
            if (spec->HasField(FieldNames::defaultValue)) return true;
        }
        return GetFallback() != nullptr;
    }
    if (spec_ && spec_->HasField(FieldNames::defaultValue)) return true;
    return GetFallback() != nullptr;
}

bool UsdAttribute::HasAuthoredValue() const {
    if (graph_ && primIndex_) {
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePath(entry, path_);
            if (sourcePath.IsEmpty()) continue;
            const auto* spec = layer.GetAttributeSpec(sourcePath);
            if (!spec) continue;
            if (spec->HasField(FieldNames::defaultValue) ||
                spec->HasField(FieldNames::timeSamples)) {
                return true;
            }
        }
        return false;
    }
    return spec_ && (spec_->HasField(FieldNames::defaultValue) ||
                     spec_->HasField(FieldNames::timeSamples));
}

const Value* UsdAttribute::GetFallback() const {
    if (hasSchemaDef_ && schemaDef_.fallback) {
        return &(*schemaDef_.fallback);
    }
    return nullptr;
}

bool UsdAttribute::HasTimeSamples() const {
    if (graph_ && primIndex_) {
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePath(entry, path_);
            if (sourcePath.IsEmpty()) continue;
            const auto* spec = layer.GetAttributeSpec(sourcePath);
            if (spec && spec->HasField(FieldNames::timeSamples)) return true;
        }
        return false;
    }
    return spec_ && spec_->HasField(FieldNames::timeSamples);
}

const Value* UsdAttribute::GetTimeSamplesField() const {
    if (graph_ && primIndex_) {
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePath(entry, path_);
            if (sourcePath.IsEmpty()) continue;
            const auto* spec = layer.GetAttributeSpec(sourcePath);
            if (!spec) continue;
            auto* field = spec->GetField(FieldNames::timeSamples);
            if (field) return field;
        }
        return nullptr;
    }
    return spec_ ? spec_->GetField(FieldNames::timeSamples) : nullptr;
}

std::vector<double> UsdAttribute::GetTimeSampleTimes() const {
    std::vector<double> times;

    if (graph_ && primIndex_) {
        // First layer with timeSamples wins, apply forward retiming
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePath(entry, path_);
            if (sourcePath.IsEmpty()) continue;
            const auto* spec = layer.GetAttributeSpec(sourcePath);
            if (!spec) continue;
            auto* tsField = spec->GetField(FieldNames::timeSamples);
            if (!tsField) continue;
            auto* dict = tsField->Get<Dictionary>();
            if (!dict) continue;
            times.reserve(dict->size());
            for (const auto& [key, _] : *dict) {
                double t = std::stod(key);
                // Apply forward retiming: composed_time = t * scale + offset
                t = t * entry.retiming.scale + entry.retiming.offset;
                times.push_back(t);
            }
            std::sort(times.begin(), times.end());
            return times;
        }
        return times;
    }

    auto* tsField = GetTimeSamplesField();
    if (!tsField) return times;
    auto* dict = tsField->Get<Dictionary>();
    if (!dict) return times;
    times.reserve(dict->size());
    for (const auto& [key, _] : *dict) {
        times.push_back(std::stod(key));
    }
    std::sort(times.begin(), times.end());
    return times;
}

Token UsdAttribute::GetColorSpace() const {
    if (graph_ && primIndex_) {
        auto authored = GetStrongestAttributeTokenField(
            *graph_, primIndex_, path_, FieldNames::colorSpace);
        return authored ? *authored : Token();
    }
    return ValueAsToken(spec_ ? spec_->GetField(FieldNames::colorSpace) : nullptr);
}

bool UsdAttribute::HasColorSpace() const {
    if (graph_ && primIndex_) {
        return HasAnyAuthoredField(*graph_, primIndex_, path_,
                                   FieldNames::colorSpace,
                                   SpecType::Attribute);
    }
    return spec_ && spec_->HasField(FieldNames::colorSpace);
}

bool UsdAttribute::SetColorSpace(const Token& colorSpace) {
    auto* spec = GetOrCreateMutableSpec();
    if (!spec) return false;
    spec->SetField(FieldNames::colorSpace, Value(colorSpace));
    spec_ = GetMutableLayer().GetAttributeSpec(path_);
    return true;
}

bool UsdAttribute::ClearColorSpace() {
    if (layerIndex_ == kNoLayer) return false;
    auto* spec = GetMutableLayer().GetSpec(path_);
    if (!spec) return false;
    spec->RemoveField(FieldNames::colorSpace);
    spec_ = GetMutableLayer().GetAttributeSpec(path_);
    return true;
}

Token UsdAttribute::ComputeColorSpaceName() const {
    if (HasColorSpace()) return GetColorSpace();
    return ComputePrimColorSpaceName(graph_, primIndex_, path_.GetPrimPath());
}

bool UsdAttribute::HasConnections() const {
    if (graph_ && primIndex_) {
        return HasAnyAuthoredField(*graph_, primIndex_, path_,
                                   FieldNames::connectionPaths,
                                   SpecType::Attribute);
    }
    return spec_ && spec_->HasField(FieldNames::connectionPaths);
}

std::vector<Path> UsdAttribute::GetConnections() const {
    std::vector<Path> result;
    if (graph_ && primIndex_) {
        // Spec §12.2.6: connection paths combine across the opinion
        // stack per the generic listop metadata rules.
        auto combined = ResolveListOpField<Path>(
            *graph_, primIndex_, path_, FieldNames::connectionPaths,
            SpecType::Attribute);
        if (combined) return combined->GetItems();
        return result;
    }
    if (!spec_) return result;
    auto* field = spec_->GetField(FieldNames::connectionPaths);
    if (!field || field->IsBlock()) return result;
    if (auto* lop = field->Get<ListOp<Path>>()) {
        return lop->GetItems();
    }
    if (auto* lop = field->Get<ListOp<std::string>>()) {
        // Legacy in-memory listop — convert via Path::Parse.
        auto items = lop->GetItems();
        result.reserve(items.size());
        for (const auto& s : items) {
            Path p = Path::Parse(s);
            if (!p.IsEmpty()) result.push_back(std::move(p));
        }
    }
    return result;
}

// ============================================================
// UsdAttribute — write operations
// ============================================================

Spec* UsdAttribute::GetOrCreateMutableSpec() {
    if (layerIndex_ == kNoLayer) return nullptr;
    auto* spec = GetMutableLayer().GetSpec(path_);
    if (spec) return spec;
    Spec newSpec(SpecType::Attribute);
    // Populate typeName from schema definition so the USDA writer
    // can emit a proper type declaration for schema-defined attributes.
    if (hasSchemaDef_ && !schemaDef_.typeName.empty()) {
        newSpec.SetTypeName(Token(schemaDef_.typeName));
    }
    GetMutableLayer().SetSpec(path_, std::move(newSpec));
    return GetMutableLayer().GetSpec(path_);
}

bool UsdAttribute::Set(Value val) {
    auto* spec = GetOrCreateMutableSpec();
    if (!spec) return false;
    ApplyRoleFromTypeName(val, GetTypeName());
    spec->SetField(FieldNames::defaultValue, std::move(val));
    spec_ = GetMutableLayer().GetAttributeSpec(path_);
    return true;
}

bool UsdAttribute::SetTimeSample(double time, Value val) {
    auto* spec = GetOrCreateMutableSpec();
    if (!spec) return false;
    ApplyRoleFromTypeName(val, GetTypeName());

    std::string timeKey;
    if (time == static_cast<int64_t>(time)) {
        timeKey = std::to_string(static_cast<int64_t>(time));
    } else {
        timeKey = std::to_string(time);
    }

    auto* existingField = spec->GetField(FieldNames::timeSamples);
    if (existingField) {
        auto* constDict = existingField->Get<Dictionary>();
        if (constDict) {
            Dictionary updated = *constDict;
            updated[timeKey] = std::move(val);
            spec->SetField(FieldNames::timeSamples, Value(std::move(updated)));
            spec_ = GetMutableLayer().GetAttributeSpec(path_);
            return true;
        }
    }

    Dictionary dict;
    dict[timeKey] = std::move(val);
    spec->SetField(FieldNames::timeSamples, Value(std::move(dict)));
    spec_ = GetMutableLayer().GetAttributeSpec(path_);
    return true;
}

bool UsdAttribute::ClearDefault() {
    if (layerIndex_ == kNoLayer) return false;
    auto* spec = GetMutableLayer().GetSpec(path_);
    if (!spec) return false;
    spec->RemoveField(FieldNames::defaultValue);
    return true;
}

bool UsdAttribute::ClearTimeSamples() {
    if (layerIndex_ == kNoLayer) return false;
    auto* spec = GetMutableLayer().GetSpec(path_);
    if (!spec) return false;
    spec->RemoveField(FieldNames::timeSamples);
    return true;
}

bool UsdAttribute::Block() {
    auto* spec = GetOrCreateMutableSpec();
    if (!spec) return false;
    spec->SetField(FieldNames::defaultValue, Value(ValueBlock{}));
    spec_ = GetMutableLayer().GetAttributeSpec(path_);
    return true;
}

// ============================================================
// UsdPrim — write operations
// ============================================================

UsdAttribute UsdPrim::CreateAttribute(const Token& name,
                                       const Token& typeName,
                                       bool custom) {
    if (layerIndex_ == kNoLayer || path_.IsEmpty()) return {};
    Path propPath = path_.AppendProperty(name);
    if (propPath.IsEmpty()) return {};

    auto* existing = GetMutableLayer().GetAttributeSpec(propPath);
    if (existing) {
        return UsdAttribute(propPath, name, existing, graph_, primIndex_, layerIndex_);
    }

    Spec attrSpec(SpecType::Attribute);
    attrSpec.SetTypeName(typeName);
    attrSpec.SetCustom(custom);
    GetMutableLayer().SetSpec(propPath, std::move(attrSpec));

    return UsdAttribute(propPath, name, GetMutableLayer().GetAttributeSpec(propPath),
                        graph_, primIndex_, layerIndex_);
}

UsdRelationship UsdPrim::CreateRelationship(const Token& name) {
    if (layerIndex_ == kNoLayer || path_.IsEmpty()) return {};
    Path propPath = path_.AppendProperty(name);
    if (propPath.IsEmpty()) return {};

    auto* existing = GetMutableLayer().GetRelationshipSpec(propPath);
    if (existing) {
        return UsdRelationship(propPath, name, existing, graph_, primIndex_, layerIndex_);
    }

    Spec relSpec(SpecType::Relationship);
    GetMutableLayer().SetSpec(propPath, std::move(relSpec));

    return UsdRelationship(propPath, name, GetMutableLayer().GetRelationshipSpec(propPath),
                           graph_, primIndex_, layerIndex_);
}

bool UsdPrim::ApplyAPI(const Token& schemaName) {
    if (layerIndex_ == kNoLayer || path_.IsEmpty()) return false;

    // Get existing apiSchemas listop, or start fresh
    auto existing = GetAuthoredAppliedSchemasForPrim(
        graph_, primIndex_, path_, spec_);
    for (const auto& s : existing) {
        if (s == schemaName) return true;  // already applied
    }

    // Build a prepend listop with the new schema. apiSchemas is spec-
    // declared listop<token> (§13.2.1.2), so storage is ListOp<Token>.
    ListOp<Token> newOp;
    newOp.SetPrependedItems({schemaName});

    // Get the spec to write to
    auto* spec = GetMutableLayer().GetPrimSpec(path_);
    if (!spec) return false;

    // Combine with existing
    auto* field = spec->GetField(FieldNames::apiSchemas);
    if (field) {
        if (auto* existingOp = field->Get<ListOp<Token>>()) {
            newOp = newOp.Combine(*existingOp);
        }
    }
    const_cast<Spec*>(spec)->SetField(FieldNames::apiSchemas,
                                       Value(std::move(newOp)));
    if (graph_) graph_->InvalidatePrimDefCache(path_.GetText());
    return true;
}

bool UsdPrim::ApplyAPI(const Token& schemaName,
                       const Token& instanceName) {
    return ApplyAPI(Token(schemaName.GetString() + ":" + instanceName.GetString()));
}

// --- UsdRelationship write operations ---

Spec* UsdRelationship::GetOrCreateMutableSpec() {
    if (layerIndex_ == kNoLayer) return nullptr;
    auto* spec = GetMutableLayer().GetSpec(path_);
    if (spec) return spec;
    Spec newSpec(SpecType::Relationship);
    GetMutableLayer().SetSpec(path_, std::move(newSpec));
    return GetMutableLayer().GetSpec(path_);
}

bool UsdRelationship::SetTargets(const std::vector<Path>& targets) {
    auto* spec = GetOrCreateMutableSpec();
    if (!spec) return false;
    ListOp<Path> op;
    op.SetExplicitItems(targets);
    spec->SetField(FieldNames::targetPaths, Value(std::move(op)));
    spec_ = GetMutableLayer().GetRelationshipSpec(path_);
    return true;
}

bool UsdRelationship::AddTarget(const Path& target) {
    auto* spec = GetOrCreateMutableSpec();
    if (!spec) return false;
    auto existing = GetTargets();
    existing.push_back(target);
    return SetTargets(existing);
}

bool UsdRelationship::ClearTargets() {
    auto* spec = GetOrCreateMutableSpec();
    if (!spec) return false;
    spec->RemoveField(FieldNames::targetPaths);
    spec_ = GetMutableLayer().GetRelationshipSpec(path_);
    return true;
}

// ============================================================
// Linear interpolation helpers (spec Section 12.5.2)
// ============================================================

namespace {

template <typename T>
T Lerp(T a, T b, double t) {
    return static_cast<T>(a + t * (b - a));
}

template <>
Half Lerp<Half>(Half a, Half b, double t) {
    float fa = static_cast<float>(a);
    float fb = static_cast<float>(b);
    return Half(fa + static_cast<float>(t) * (fb - fa));
}

template <typename T, size_t N>
Vec<T, N> LerpVec(const Vec<T, N>& a, const Vec<T, N>& b, double t) {
    Vec<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = Lerp(a[i], b[i], t);
    }
    return result;
}

template <typename T, size_t R, size_t C>
Matrix<T, R, C> LerpMatrix(const Matrix<T, R, C>& a, const Matrix<T, R, C>& b, double t) {
    Matrix<T, R, C> result;
    for (size_t i = 0; i < R * C; ++i) {
        result.data[i] = Lerp(a.data[i], b.data[i], t);
    }
    return result;
}

template <typename T>
double ToDouble(T v) { return static_cast<double>(v); }
inline double ToDouble(Half v) { return static_cast<double>(static_cast<float>(v)); }

template <typename T>
T FromDouble(double v);
template <> inline float FromDouble<float>(double v) { return static_cast<float>(v); }
template <> inline double FromDouble<double>(double v) { return v; }
template <> inline Half FromDouble<Half>(double v) { return Half(static_cast<float>(v)); }

template <typename T>
Quat<T> SlerpQuat(const Quat<T>& a, const Quat<T>& b, double t) {
    double dot = 0;
    for (int i = 0; i < 4; ++i) {
        dot += ToDouble(a[i]) * ToDouble(b[i]);
    }

    Quat<T> b2 = b;
    if (dot < 0) {
        dot = -dot;
        for (int i = 0; i < 4; ++i) b2[i] = FromDouble<T>(-ToDouble(b[i]));
    }

    if (dot > 0.9995) {
        Quat<T> result;
        for (int i = 0; i < 4; ++i) {
            result[i] = Lerp(a[i], b2[i], t);
        }
        double len = 0;
        for (int i = 0; i < 4; ++i) len += ToDouble(result[i]) * ToDouble(result[i]);
        len = std::sqrt(len);
        if (len > 0) {
            for (int i = 0; i < 4; ++i) result[i] = FromDouble<T>(ToDouble(result[i]) / len);
        }
        return result;
    }

    double theta = std::acos(dot);
    double sinTheta = std::sin(theta);
    double wa = std::sin((1.0 - t) * theta) / sinTheta;
    double wb = std::sin(t * theta) / sinTheta;

    Quat<T> result;
    for (int i = 0; i < 4; ++i) {
        result[i] = FromDouble<T>(wa * ToDouble(a[i]) + wb * ToDouble(b2[i]));
    }
    return result;
}

Value PreserveInterpolatedRole(Value result, const Value& lower, const Value& upper) {
    if (lower.GetRole() != Role::None && lower.GetRole() == upper.GetRole()) {
        result.SetRole(lower.GetRole());
    }
    return result;
}

Value LinearInterpolate(const Value& lower, const Value& upper, double t) {
    if (auto* a = lower.Get<Half>()) {
        if (auto* b = upper.Get<Half>())
            return PreserveInterpolatedRole(Value(Lerp(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<Float>()) {
        if (auto* b = upper.Get<Float>())
            return PreserveInterpolatedRole(Value(Lerp(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<Double>()) {
        if (auto* b = upper.Get<Double>())
            return PreserveInterpolatedRole(Value(Lerp(*a, *b, t)), lower, upper);
    }
    if (lower.GetTypeId() == TypeId::TimeCode) {
        auto* a = lower.Get<TimeCode>();
        auto* b = upper.Get<TimeCode>();
        if (a && b)
            return PreserveInterpolatedRole(Value(Lerp(*a, *b, t), nullptr), lower, upper);
    }
    if (auto* a = lower.Get<GfVec2h>()) {
        if (auto* b = upper.Get<GfVec2h>())
            return PreserveInterpolatedRole(Value(LerpVec(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<GfVec3h>()) {
        if (auto* b = upper.Get<GfVec3h>())
            return PreserveInterpolatedRole(Value(LerpVec(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<GfVec4h>()) {
        if (auto* b = upper.Get<GfVec4h>())
            return PreserveInterpolatedRole(Value(LerpVec(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<GfVec2f>()) {
        if (auto* b = upper.Get<GfVec2f>())
            return PreserveInterpolatedRole(Value(LerpVec(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<GfVec3f>()) {
        if (auto* b = upper.Get<GfVec3f>())
            return PreserveInterpolatedRole(Value(LerpVec(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<GfVec4f>()) {
        if (auto* b = upper.Get<GfVec4f>())
            return PreserveInterpolatedRole(Value(LerpVec(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<GfVec2d>()) {
        if (auto* b = upper.Get<GfVec2d>())
            return PreserveInterpolatedRole(Value(LerpVec(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<GfVec3d>()) {
        if (auto* b = upper.Get<GfVec3d>())
            return PreserveInterpolatedRole(Value(LerpVec(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<GfVec4d>()) {
        if (auto* b = upper.Get<GfVec4d>())
            return PreserveInterpolatedRole(Value(LerpVec(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<GfMatrix2d>()) {
        if (auto* b = upper.Get<GfMatrix2d>())
            return PreserveInterpolatedRole(Value(LerpMatrix(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<GfMatrix3d>()) {
        if (auto* b = upper.Get<GfMatrix3d>())
            return PreserveInterpolatedRole(Value(LerpMatrix(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<GfMatrix4d>()) {
        if (auto* b = upper.Get<GfMatrix4d>())
            return PreserveInterpolatedRole(Value(LerpMatrix(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<GfQuath>()) {
        if (auto* b = upper.Get<GfQuath>())
            return PreserveInterpolatedRole(Value(SlerpQuat(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<GfQuatf>()) {
        if (auto* b = upper.Get<GfQuatf>())
            return PreserveInterpolatedRole(Value(SlerpQuat(*a, *b, t)), lower, upper);
    }
    if (auto* a = lower.Get<GfQuatd>()) {
        if (auto* b = upper.Get<GfQuatd>())
            return PreserveInterpolatedRole(Value(SlerpQuat(*a, *b, t)), lower, upper);
    }
    return {};
}

} // end anonymous namespace

UsdResolvedValue ResolveTimeSample(const Dictionary& samples, double time,
                                   UsdInterpolationType interp) {
    if (samples.empty()) return {};

    struct Sample { double time; const Value* value; };
    std::vector<Sample> sorted;
    sorted.reserve(samples.size());
    for (const auto& [key, val] : samples) {
        sorted.push_back({std::stod(key), &val});
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const Sample& a, const Sample& b) { return a.time < b.time; });

    if (time <= sorted.front().time) {
        const Value* v = sorted.front().value;
        if (v->IsBlock()) return {};
        return {*v, true};
    }
    if (time >= sorted.back().time) {
        const Value* v = sorted.back().value;
        if (v->IsBlock()) return {};
        return {*v, true};
    }

    for (size_t i = 0; i + 1 < sorted.size(); ++i) {
        if (time >= sorted[i].time && time <= sorted[i + 1].time) {
            if (time == sorted[i].time) {
                if (sorted[i].value->IsBlock()) return {};
                return {*sorted[i].value, true};
            }
            if (time == sorted[i + 1].time) {
                if (sorted[i + 1].value->IsBlock()) return {};
                return {*sorted[i + 1].value, true};
            }

            if (sorted[i].value->IsBlock() || sorted[i + 1].value->IsBlock()) {
                return {};
            }

            if (interp == UsdInterpolationType::Held) {
                return {*sorted[i].value, true};
            }

            double fraction = (time - sorted[i].time) /
                              (sorted[i + 1].time - sorted[i].time);
            Value interpolated = LinearInterpolate(*sorted[i].value,
                                                   *sorted[i + 1].value,
                                                   fraction);
            if (!interpolated.IsEmpty()) {
                return {std::move(interpolated), true};
            }
            return {*sorted[i].value, true};
        }
    }

    return {};
}

namespace {

// Remap time samples in a Dictionary according to a retiming
Dictionary RemapTimeSamples(const Dictionary& samples, const Retiming& retiming) {
    if (retiming.offset == 0.0 && retiming.scale == 1.0) return samples;
    Dictionary remapped;
    for (const auto& [timeStr, value] : samples) {
        double t = std::stod(timeStr);
        double newTime = t * retiming.scale + retiming.offset;
        std::string newKey;
        if (newTime == static_cast<int64_t>(newTime)) {
            newKey = std::to_string(static_cast<int64_t>(newTime));
        } else {
            newKey = std::to_string(newTime);
        }
        remapped[newKey] = value;
    }
    return remapped;
}

// Resolve an asset-typed Value in-place using the given anchor layer path and resolver.
// Returns true if the value was modified.
static bool ResolveAssetValue(Value& val, const std::string& anchorLayerPath,
                               const AssetResolver& resolver) {
    if (!resolver) return false;
    if (val.GetTypeId() == TypeId::Asset && !val.IsArray()) {
        auto* s = val.Get<std::string>();
        if (s && !s->empty()) {
            std::string resolved = resolver(anchorLayerPath, *s);
            if (resolved != *s) {
                val = Value(Value::AssetTag{}, std::move(resolved));
                return true;
            }
        }
    } else if (val.GetTypeId() == TypeId::Asset && val.IsArray()) {
        auto* arr = val.Get<std::vector<std::string>>();
        if (arr) {
            bool modified = false;
            for (auto& s : *arr) {
                if (!s.empty()) {
                    std::string res = resolver(anchorLayerPath, s);
                    if (res != s) {
                        s = std::move(res);
                        modified = true;
                    }
                }
            }
            return modified;
        }
    }
    return false;
}

// Recursively resolve asset path values within a Dictionary (spec §5).
// Walks all key/value pairs; for Asset-typed values, resolves in place.
// For nested Dictionary values, recurses. Returns true if anything was modified.
static bool ResolveAssetValuesInDict(Dictionary& dict, const std::string& anchorLayerPath,
                                      const AssetResolver& resolver) {
    if (!resolver) return false;
    bool modified = false;
    for (auto& [key, val] : dict) {
        if (val.GetTypeId() == TypeId::Asset) {
            if (ResolveAssetValue(val, anchorLayerPath, resolver))
                modified = true;
        } else if (val.GetTypeId() == TypeId::Dictionary) {
            auto* nested = val.Get<Dictionary>();
            if (nested && ResolveAssetValuesInDict(*nested, anchorLayerPath, resolver))
                modified = true;
        }
    }
    return modified;
}

// Resolve asset path values in any Value — handles Asset, Asset[], and Dictionary.
// Used for metadata fields which may be any of these types (spec §5).
static bool ResolveAssetValueOrDict(Value& val, const std::string& anchorLayerPath,
                                     const AssetResolver& resolver) {
    if (!resolver) return false;
    if (val.GetTypeId() == TypeId::Asset) {
        return ResolveAssetValue(val, anchorLayerPath, resolver);
    } else if (val.GetTypeId() == TypeId::Dictionary) {
        auto* dict = val.Get<Dictionary>();
        if (dict) return ResolveAssetValuesInDict(*dict, anchorLayerPath, resolver);
    }
    return false;
}

// Spec §12.2.5 — combine a Dictionary-typed prim metadata field across the
// opinion stack per the §6.6.2.1 combining rule.
//
// Walks opinion entries weakest→strongest (reverse of primIndex->entries,
// which is strongest-first after the LIVRPS sort), folding each authored
// dict on top of the accumulated weaker result via CombineDicts.
//
// Asset-path resolution is applied PER OPINION before the combine, because
// each opinion's asset paths anchor to that opinion's layer — once dicts
// merge, per-key provenance is lost. The resolver is optional; when null,
// the combine runs on the raw authored dicts.
//
// Returns std::nullopt if no opinion authored a Dictionary value for the
// field.
static std::optional<Value> ResolveDictionaryField(
        const CompositionGraph* graph,
        const PrimIndex* primIndex,
        const Path& primPath,
        const Token& fieldName,
        const AssetResolver& resolver) {
    if (!graph || !primIndex) return std::nullopt;
    const auto& entries = primIndex->entries;
    std::optional<Dictionary> accumulator;
    for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
        const auto& entry = *it;
        const Layer& layer = graph->GetLayer(entry.layerIndex);
        Path sourcePath = OpinionSourcePrimPath(entry, primPath);
        if (sourcePath.IsEmpty()) continue;
        const auto* spec = layer.GetPrimSpec(sourcePath);
        if (!spec) continue;
        auto* fieldVal = spec->GetField(fieldName);
        if (!fieldVal) continue;
        const auto* dict = fieldVal->Get<Dictionary>();
        if (!dict) continue;
        // Resolve asset paths per-opinion against this entry's layer.
        Dictionary contribution = *dict;
        if (resolver) {
            ResolveAssetValuesInDict(contribution,
                                     graph->layerPaths[entry.layerIndex],
                                     resolver);
        }
        if (!accumulator) {
            accumulator = std::move(contribution);
        } else {
            // Iterating reverse, so `contribution` is the next STRONGER
            // opinion relative to everything already folded in.
            accumulator = CombineDicts(contribution, *accumulator);
        }
    }
    if (!accumulator) return std::nullopt;
    return Value(std::move(*accumulator));
}

static std::vector<Path> MapPathListItemsToTarget(
        const std::vector<Path>& items,
        const PathMappingPtr& mapping) {
    if (!mapping || mapping->isIdentity) return items;

    std::vector<Path> mapped;
    mapped.reserve(items.size());
    for (const auto& item : items) {
        if (item.IsEmpty()) {
            mapped.push_back(item);
            continue;
        }
        Path targetPath = mapping->MapToTarget(item);
        if (!targetPath.IsEmpty())
            mapped.push_back(std::move(targetPath));
    }
    return mapped;
}

static ListOp<Path> MapPathListOpToTarget(
        const ListOp<Path>& lop,
        const PathMappingPtr& mapping) {
    if (!mapping || mapping->isIdentity) return lop;
    if (lop.IsExplicit()) {
        return ListOp<Path>::CreateExplicit(
            MapPathListItemsToTarget(lop.GetExplicitItems(), mapping));
    }
    return ListOp<Path>::CreateComposable(
        MapPathListItemsToTarget(lop.GetPrependedItems(), mapping),
        MapPathListItemsToTarget(lop.GetAppendedItems(), mapping),
        MapPathListItemsToTarget(lop.GetDeletedItems(), mapping));
}

// Spec §12.2.6 — combine a generic listop metadata field across the
// opinion stack per §6.6.3.6 (ListOp::Combine handles the per-pair
// rules, including pruning at first explicit).
//
// The function is templated on the listop element type so the same
// fold works for ListOp<Path> (relationships, connections), for
// ListOp<std::string> (apiSchemas), and — when added — for the other
// spec-required element types (int, int64, uint, uint64, token).
//
// Walks opinion entries strongest-first. The accumulator starts at
// the strongest authored listop, then weaker authored listops fold
// in via accumulator->Combine(weaker). A ValueBlock at any point
// stops further accumulation per §7.6.4.2.1: weaker opinions are
// blocked and may not contribute. specType disambiguates the source
// spec form (Prim / Attribute / Relationship), since the same
// fieldName lives on different spec types depending on the field.
template <typename T>
static std::optional<ListOp<T>> ResolveListOpField(
        const CompositionGraph& graph,
        const PrimIndex* primIndex,
        const Path& path,
        const Token& fieldName,
        SpecType specType) {
    if (!primIndex) return std::nullopt;
    std::optional<ListOp<T>> accumulator;
    for (const auto& entry : primIndex->entries) {
        const Layer& layer = graph.GetLayer(entry.layerIndex);
        Path sourcePath = OpinionSourcePath(entry, path);
        if (sourcePath.IsEmpty()) continue;
        const Spec* spec = nullptr;
        switch (specType) {
            case SpecType::Prim:
            case SpecType::Variant:
                spec = layer.GetPrimSpec(sourcePath); break;
            case SpecType::Attribute:
                spec = layer.GetAttributeSpec(sourcePath); break;
            case SpecType::Relationship:
                spec = layer.GetRelationshipSpec(sourcePath); break;
            default:
                continue;
        }
        if (!spec) continue;
        const auto* fieldVal = spec->GetField(fieldName);
        if (!fieldVal) continue;
        if (fieldVal->IsBlock()) {
            // ValueBlock: weaker opinions are blocked. Stop folding.
            break;
        }
        const auto* lop = fieldVal->Get<ListOp<T>>();
        if (!lop) continue;
        ListOp<T> contribution = *lop;
        if constexpr (std::is_same_v<T, Path>) {
            contribution = MapPathListOpToTarget(*lop, entry.pathMapping);
        }
        if (!accumulator) {
            accumulator = std::move(contribution);
        } else {
            // accumulator is stronger; lop is the next weaker contribution.
            accumulator = accumulator->Combine(contribution);
        }
    }
    return accumulator;
}

// True if any opinion in the stack authored the named field on a
// spec of the given form. Used by UsdAttribute::HasConnections /
// UsdRelationship::HasTargets to compose the "is anything authored
// anywhere" predicate, replacing the pre-composition single-spec
// check.
static bool HasAnyAuthoredField(
        const CompositionGraph& graph,
        const PrimIndex* primIndex,
        const Path& path,
        const Token& fieldName,
        SpecType specType) {
    if (!primIndex) return false;
    for (const auto& entry : primIndex->entries) {
        const Layer& layer = graph.GetLayer(entry.layerIndex);
        Path sourcePath = OpinionSourcePath(entry, path);
        if (sourcePath.IsEmpty()) continue;
        const Spec* spec = nullptr;
        switch (specType) {
            case SpecType::Prim:
            case SpecType::Variant:
                spec = layer.GetPrimSpec(sourcePath); break;
            case SpecType::Attribute:
                spec = layer.GetAttributeSpec(sourcePath); break;
            case SpecType::Relationship:
                spec = layer.GetRelationshipSpec(sourcePath); break;
            default:
                continue;
        }
        if (spec && spec->HasField(fieldName)) return true;
    }
    return false;
}

static Token ValueAsToken(const Value* value) {
    if (!value) return {};
    if (auto* token = value->Get<Token>()) return *token;
    return {};
}

static std::optional<Token> GetStrongestAttributeTokenField(
        const CompositionGraph& graph,
        const PrimIndex* primIndex,
        const Path& attrPath,
        const Token& fieldName) {
    if (!primIndex) return std::nullopt;
    for (const auto& entry : primIndex->entries) {
        const Layer& layer = graph.GetLayer(entry.layerIndex);
        Path sourcePath = OpinionSourcePath(entry, attrPath);
        if (sourcePath.IsEmpty()) continue;
        const auto* spec = layer.GetAttributeSpec(sourcePath);
        if (!spec) continue;
        if (const auto* field = spec->GetField(fieldName)) {
            return ValueAsToken(field);
        }
    }
    return std::nullopt;
}

static bool HasAppliedColorSpaceAPI(const CompositionGraph* graph,
                                    const PrimIndex* primIndex,
                                    const Path& primPath,
                                    const Spec* specFallback) {
    static const Token colorSpaceAPI("ColorSpaceAPI");
    if (graph && primIndex) {
        auto combined = ResolveListOpField<Token>(
            *graph, primIndex, primPath, FieldNames::apiSchemas, SpecType::Prim);
        if (!combined) return false;
        auto schemas = combined->GetItems();
        return std::find(schemas.begin(), schemas.end(), colorSpaceAPI) != schemas.end();
    }
    if (!specFallback) return false;
    const auto* field = specFallback->GetField(FieldNames::apiSchemas);
    const auto* listOp = field ? field->Get<ListOp<Token>>() : nullptr;
    if (!listOp) return false;
    auto schemas = listOp->GetItems();
    return std::find(schemas.begin(), schemas.end(), colorSpaceAPI) != schemas.end();
}

static Token ComputePrimColorSpaceName(const CompositionGraph* graph,
                                       const PrimIndex* primIndex,
                                       const Path& primPath) {
    if (primPath.IsEmpty()) return {};

    if (!graph || !primIndex) return {};

    Path path = primPath;
    while (!path.IsEmpty() && !path.IsAbsoluteRoot()) {
        const PrimIndex* idx = (path == primPath) ? primIndex : graph->GetPrimIndex(path);
        if (idx && HasAppliedColorSpaceAPI(graph, idx, path, nullptr)) {
            auto authored = GetStrongestAttributeTokenField(
                *graph, idx, path.AppendProperty(Token("colorSpace:name")),
                FieldNames::defaultValue);
            if (authored) return *authored;
        }
        path = path.GetParentPath();
    }
    return {};
}

} // anonymous namespace

// ============================================================
// UsdAttribute — Value Resolution (spec Section 12.3)
// ============================================================

UsdResolvedValue UsdAttribute::Get(UsdTimeCode time) const {
    return Get(time, UsdInterpolationType::Linear);
}

UsdResolvedValue UsdAttribute::Get(UsdTimeCode time, UsdInterpolationType interp) const {
    // Spec Section 12.3: uniform attributes always use Held interpolation
    if (GetVariability() == Variability::Uniform)
        interp = UsdInterpolationType::Held;

    if (time.IsDefault()) {
        // Default time query — walk opinion stack for default value
        if (graph_ && primIndex_) {
            for (const auto& entry : primIndex_->entries) {
                const Layer& layer = graph_->GetLayer(entry.layerIndex);
                Path sourcePath = OpinionSourcePath(entry, path_);
                if (sourcePath.IsEmpty()) continue;
                const auto* spec = layer.GetAttributeSpec(sourcePath);
                if (!spec) continue;
                auto* authored = spec->GetField(FieldNames::defaultValue);
                if (authored) {
                    if (authored->IsBlock()) {
                        auto* fb = GetFallback();
                        if (fb) return {*fb, true};
                        return {};
                    }
                    if (authored->GetTypeId() == TypeId::Asset && graph_->resolver) {
                        Value copy = *authored;
                        ResolveAssetValue(copy, graph_->layerPaths[entry.layerIndex], graph_->resolver);
                        return {std::move(copy), true};
                    }
                    return {*authored, true};
                }
            }
            auto* fb = GetFallback();
            if (fb) return {*fb, true};
            return {};
        }

        // Legacy single-layer path
        if (spec_) {
            auto* authored = spec_->GetField(FieldNames::defaultValue);
            if (authored) {
                if (authored->IsBlock()) {
                    auto* fb = GetFallback();
                    if (fb) return {*fb, true};
                    return {};
                }
                return {*authored, true};
            }
        }
        auto* fb = GetFallback();
        if (fb) return {*fb, true};
        return {};
    }

    // Time-varying query — single per-opinion walk (spec §12.3, §12.3.4.5).
    // Strongest opinion first; within each opinion the field priority is
    // timeSamples > spline > default > clips. Spline evaluation isn't
    // implemented yet (the slot is reserved). "Clip data is just weaker
    // than Local" (§12.3.4.5): an opinion's locally authored value beats
    // any clips authored at the same opinion.
    if (graph_ && primIndex_) {
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePath(entry, path_);
            if (sourcePath.IsEmpty()) continue;

            // Apply reverse retiming once per entry — timeSamples and
            // clip time lookups both use queryTime.
            double queryTime = time.GetValue();
            if (entry.retiming.scale != 0.0) {
                queryTime = (queryTime - entry.retiming.offset) / entry.retiming.scale;
            }

            // Priority 1: timeSamples on the attribute at this layer.
            // Priority 2: spline on the attribute at this layer.
            // Priority 3: default on the attribute at this layer.
            const auto* spec = layer.GetAttributeSpec(sourcePath);
            if (spec) {
                auto* tsField = spec->GetField(FieldNames::timeSamples);
                if (tsField) {
                    if (tsField->IsBlock()) {
                        auto* fb = GetFallback();
                        if (fb) return {*fb, true};
                        return {};
                    }
                    if (auto* dict = tsField->Get<Dictionary>()) {
                        if (!dict->empty()) {
                            auto resolved = ResolveTimeSample(*dict, queryTime, interp);
                            if (resolved.found) {
                                if (resolved.value.GetTypeId() == TypeId::Asset && graph_->resolver) {
                                    ResolveAssetValue(resolved.value, graph_->layerPaths[entry.layerIndex], graph_->resolver);
                                }
                                return resolved;
                            }
                            auto* fb = GetFallback();
                            if (fb) return {*fb, true};
                            return {};
                        }
                    }
                }
                // Priority 2: spline (§12.5.3). A `none` segment or
                // a `none` extrapolation boundary returns nullopt and
                // falls through to the opinion's default below — per
                // §7.4.2.4.2 the `none` interp mode represents an
                // explicit value block across that segment.
                //
                // Spline storage is always double (spec §7.4.2.4 allows
                // half/float/double; we widen at parse). Narrow back to
                // the attribute's authored typeName before returning so
                // callers that do Get<Float>() / Get<Half>() see the
                // expected concrete type.
                auto* splineField = spec->GetField(FieldNames::spline);
                if (splineField && !splineField->IsBlock()) {
                    if (const auto* s = splineField->Get<Spline>()) {
                        auto v = EvaluateSpline(*s, queryTime);
                        if (v) {
                            if (auto* d = v->Get<Double>()) {
                                if (auto* tn = spec->GetField(FieldNames::typeName)) {
                                    if (auto* tok = tn->Get<Token>()) {
                                        const auto& name = tok->GetString();
                                        if (name == "float")
                                            return {Value(static_cast<Float>(*d)), true};
                                        if (name == "half")
                                            return {Value(Half(static_cast<float>(*d))), true};
                                    }
                                }
                            }
                            return {std::move(*v), true};
                        }
                    }
                }
                auto* authored = spec->GetField(FieldNames::defaultValue);
                if (authored) {
                    if (authored->IsBlock()) {
                        auto* fb = GetFallback();
                        if (fb) return {*fb, true};
                        return {};
                    }
                    if (authored->GetTypeId() == TypeId::Asset && graph_->resolver) {
                        Value copy = *authored;
                        ResolveAssetValue(copy, graph_->layerPaths[entry.layerIndex], graph_->resolver);
                        return {std::move(copy), true};
                    }
                    return {*authored, true};
                }
            }

            // Priority 4: clips authored on the PRIM at this layer.
            // Per §12.3.4.5, clips contribute attribute values only
            // when nothing local on the attribute has already spoken
            // at this opinion.
            const Spec* primSpec = layer.GetPrimSpec(sourcePath.GetPrimPath());
            if (primSpec && primSpec->GetField(FieldNames::clips)) {
                if (!graph_->clipLayerCache) {
                    graph_->clipLayerCache = std::make_shared<ClipLayerCache>();
                }
                auto clipSets = MaterializeClipSetsForLayer(
                    *primSpec,
                    graph_->layerPaths[entry.layerIndex],
                    entry.layerIndex,
                    graph_->resolver);
                Path stagePrimPath = sourcePath.GetPrimPath();
                for (const auto& clipSet : clipSets) {
                    auto v = EvaluateClipSetAtTime(clipSet, sourcePath,
                                                    stagePrimPath, queryTime,
                                                    *graph_->clipLayerCache);
                    if (v) {
                        if (v->GetTypeId() == TypeId::Asset && graph_->resolver) {
                            ResolveAssetValue(*v, graph_->layerPaths[entry.layerIndex], graph_->resolver);
                        }
                        return {std::move(*v), true};
                    }
                }
            }
        }
        auto* fb = GetFallback();
        if (fb) return {*fb, true};
        return {};
    }

    // Legacy single-layer path
    if (spec_) {
        auto* tsField = spec_->GetField(FieldNames::timeSamples);
        if (tsField) {
            if (tsField->IsBlock()) {
                auto* fb = GetFallback();
                if (fb) return {*fb, true};
                return {};
            }
            auto* dict = tsField->Get<Dictionary>();
            if (dict && !dict->empty()) {
                auto resolved = ResolveTimeSample(*dict, time.GetValue(), interp);
                if (resolved.found) return resolved;
                auto* fb = GetFallback();
                if (fb) return {*fb, true};
                return {};
            }
        }
    }

    if (spec_) {
        auto* authored = spec_->GetField(FieldNames::defaultValue);
        if (authored) {
            if (authored->IsBlock()) {
                auto* fb = GetFallback();
                if (fb) return {*fb, true};
                return {};
            }
            return {*authored, true};
        }
    }

    auto* fb = GetFallback();
    if (fb) return {*fb, true};
    return {};
}

// ============================================================
// UsdRelationship
// ============================================================

bool UsdRelationship::HasTargets() const {
    if (graph_ && primIndex_) {
        return HasAnyAuthoredField(*graph_, primIndex_, path_,
                                   FieldNames::targetPaths,
                                   SpecType::Relationship);
    }
    return spec_ && spec_->HasField(FieldNames::targetPaths);
}

std::vector<Path> UsdRelationship::GetTargets() const {
    std::vector<Path> result;
    if (graph_ && primIndex_) {
        // Spec §12.2.6: relationship targets follow generic listop
        // metadata combining across the opinion stack.
        auto combined = ResolveListOpField<Path>(
            *graph_, primIndex_, path_, FieldNames::targetPaths,
            SpecType::Relationship);
        if (combined) return combined->GetItems();
        return result;
    }
    // Single-spec fallback (legacy mode): no composition graph.
    if (!spec_) return result;
    auto* field = spec_->GetField(FieldNames::targetPaths);
    if (!field || field->IsBlock()) return result;
    if (auto* lop = field->Get<ListOp<Path>>()) {
        return lop->GetItems();
    }
    // Pre-rename in-memory layers may still hold ListOp<std::string>;
    // convert through Path::Parse so callers always see typed paths.
    if (auto* lop = field->Get<ListOp<std::string>>()) {
        auto items = lop->GetItems();
        result.reserve(items.size());
        for (const auto& s : items) {
            Path p = Path::Parse(s);
            if (!p.IsEmpty()) result.push_back(std::move(p));
        }
    }
    return result;
}

std::vector<Path> UsdRelationship::GetForwardedTargets() const {
    std::vector<Path> result;
    const Layer* resolveLayer = nullptr;
    if (graph_ && layerIndex_ < graph_->GetNumLayers()) {
        resolveLayer = &graph_->GetLayer(layerIndex_);
    }
    if (!resolveLayer) return GetTargets();

    auto rawTargets = GetTargets();
    for (const auto& target : rawTargets) {
        std::string targetText = target.GetText();

        auto dotPos = targetText.find('.');
        if (dotPos == std::string::npos) {
            result.push_back(target);
            continue;
        }

        const auto* targetSpec = resolveLayer->GetSpec(target);
        if (!targetSpec || targetSpec->GetType() != SpecType::Relationship) {
            result.push_back(target);
            continue;
        }

        std::string propName = targetText.substr(dotPos + 1);
        const PrimIndex* targetPrimIndex =
            graph_ ? graph_->GetPrimIndex(target.GetPrimPath()) : nullptr;
        UsdRelationship subRel = targetPrimIndex
            ? UsdRelationship(target, Token(propName), targetSpec,
                              graph_, targetPrimIndex, layerIndex_)
            : UsdRelationship(target, Token(propName), targetSpec, layerIndex_);
        auto forwarded = subRel.GetForwardedTargets();
        result.insert(result.end(), forwarded.begin(), forwarded.end());
    }
    return result;
}

// ============================================================
// UsdPrim — graph-aware field resolution helpers
// ============================================================

namespace {

// Walk opinion stack for a prim field, returning first authored value.
const Value* GetFirstPrimField(const CompositionGraph* graph,
                                const PrimIndex* primIndex,
                                const Path& primPath,
                                const Token& fieldName) {
    if (!graph || !primIndex) return nullptr;
    for (const auto& entry : primIndex->entries) {
        const Layer& layer = graph->GetLayer(entry.layerIndex);
        Path sourcePath = OpinionSourcePrimPath(entry, primPath);
        if (sourcePath.IsEmpty()) continue;
        const auto* spec = layer.GetPrimSpec(sourcePath);
        if (!spec) continue;
        auto* val = spec->GetField(fieldName);
        if (val) return val;
    }
    return nullptr;
}

const Spec* GetStrongestPrimSpec(const CompositionGraph* graph,
                                 const PrimIndex* primIndex,
                                 const Path& primPath) {
    if (!graph || !primIndex) return nullptr;
    for (const auto& entry : primIndex->entries) {
        const Layer& layer = graph->GetLayer(entry.layerIndex);
        Path sourcePath = OpinionSourcePrimPath(entry, primPath);
        if (sourcePath.IsEmpty()) continue;
        const Spec* spec = layer.GetPrimSpec(sourcePath);
        if (spec) return spec;
    }
    return nullptr;
}

bool IsActiveValueEnabled(const Value* activeVal) {
    if (!activeVal) return true;
    if (auto* b = activeVal->Get<bool>()) return *b;
    if (auto* i = activeVal->Get<Int>()) return *i != 0;
    return true;
}

bool ApplyOrder(std::vector<Token>& childNames, const Value* orderField) {
    if (!orderField) return false;
    const auto* orderVec = orderField->Get<std::vector<std::string>>();
    if (!orderVec) return false;
    if (orderVec->empty()) return false;

    std::vector<Token> ordered;
    std::unordered_set<Token, Token::Hash> seen;
    ordered.reserve(childNames.size());
    for (const auto& name : *orderVec) {
        Token t(name);
        if (std::find(childNames.begin(), childNames.end(), t) !=
                childNames.end() &&
            seen.insert(t).second) {
            ordered.push_back(t);
        }
    }
    for (const auto& name : childNames) {
        if (seen.insert(name).second) {
            ordered.push_back(name);
        }
    }
    childNames = std::move(ordered);
    return true;
}

void SortByPathElementOrder(std::vector<Token>& names) {
    std::sort(names.begin(), names.end(), PathElementTokenLess);
}

bool SortAndApplyComposedPrimOrder(std::vector<Token>& childNames,
                                   const CompositionGraph* graph,
                                   const PrimIndex* primIndex,
                                   const Path& primPath,
                                   const Spec* fallbackSpec) {
    bool applied = false;
    if (graph && primIndex) {
        for (auto it = primIndex->entries.rbegin();
             it != primIndex->entries.rend(); ++it) {
            const auto& entry = *it;
            const Layer& layer = graph->GetLayer(entry.layerIndex);
            Path sourcePath = entry.pathMapping->isIdentity ? primPath :
                OpinionSourcePrimPath(entry, primPath);
            if (sourcePath.IsEmpty()) continue;
            const auto* spec = layer.GetPrimSpec(sourcePath);
            if (!spec) continue;
            applied = ApplyOrder(childNames,
                                 spec->GetField(FieldNames::primOrder)) ||
                      applied;
        }
        return applied;
    }

    return ApplyOrder(
        childNames,
        fallbackSpec ? fallbackSpec->GetField(FieldNames::primOrder) :
                       nullptr);
}

const Value* GetStrongestPropertyOrderField(const CompositionGraph* graph,
                                            const PrimIndex* primIndex,
                                            const Path& primPath,
                                            const Spec* fallbackSpec) {
    if (graph && primIndex) {
        if (const Value* v = GetFirstPrimField(
                graph, primIndex, primPath, FieldNames::propertyOrder)) {
            return v;
        }
    }
    return fallbackSpec ? fallbackSpec->GetField(FieldNames::propertyOrder) :
        nullptr;
}

void SortAndApplyPropertyOrder(std::vector<Token>& names,
                               const CompositionGraph* graph,
                               const PrimIndex* primIndex,
                               const Path& primPath,
                               const Spec* fallbackSpec) {
    SortByPathElementOrder(names);
    ApplyOrder(names, GetStrongestPropertyOrderField(
        graph, primIndex, primPath, fallbackSpec));
}

// Walk opinion stack for a prim field, returning first authored value
// along with the layer index for asset path resolution (spec §5).
struct PrimFieldWithProvenance {
    const Value* value = nullptr;
    size_t layerIndex = 0;
};

PrimFieldWithProvenance GetFirstPrimFieldWithProvenance(
    const CompositionGraph* graph,
    const PrimIndex* primIndex,
    const Path& primPath,
    const Token& fieldName) {
    if (!graph || !primIndex) return {};
    for (const auto& entry : primIndex->entries) {
        const Layer& layer = *graph->layers[entry.layerIndex];
        Path sourcePath = OpinionSourcePrimPath(entry, primPath);
        if (sourcePath.IsEmpty()) continue;
        const auto* spec = layer.GetPrimSpec(sourcePath);
        if (!spec) continue;
        auto* val = spec->GetField(fieldName);
        if (val) return {val, entry.layerIndex};
    }
    return {};
}

bool IsDefiningSpecifier(Specifier spec) {
    return spec == Specifier::Def || spec == Specifier::Class;
}

// Get specifier: first non-Over wins
Specifier GetComposedSpecifier(const CompositionGraph* graph,
                                const PrimIndex* primIndex,
                                const Path& primPath,
                                const Spec* fallbackSpec) {
    if (graph && primIndex) {
        for (const auto& entry : primIndex->entries) {
            const Layer& layer = graph->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePrimPath(entry, primPath);
            if (sourcePath.IsEmpty()) continue;
            const auto* spec = layer.GetPrimSpec(sourcePath);
            if (!spec) continue;
            if (spec->GetSpecifier() != Specifier::Over) {
                return spec->GetSpecifier();
            }
        }
        return Specifier::Over;
    }
    return fallbackSpec ? fallbackSpec->GetSpecifier() : Specifier::Over;
}

bool PrimAndAncestorsAreActive(const CompositionGraph* graph,
                               const PrimIndex* primIndex,
                               const Path& primPath,
                               const Spec* fallbackSpec) {
    if (!graph) return fallbackSpec ? fallbackSpec->GetActive() : true;

    Path path = primPath;
    while (!path.IsEmpty() && !path.IsAbsoluteRoot()) {
        const PrimIndex* idx =
            (path == primPath) ? primIndex : graph->GetPrimIndex(path);
        const Value* activeVal =
            GetFirstPrimField(graph, idx, path, FieldNames::active);
        if (!IsActiveValueEnabled(activeVal)) return false;
        path = path.GetParentPath();
    }
    return true;
}

bool PrimAndAncestorsAreDefined(const CompositionGraph* graph,
                                const PrimIndex* primIndex,
                                const Path& primPath,
                                const Spec* fallbackSpec) {
    if (!graph) {
        return fallbackSpec &&
               IsDefiningSpecifier(fallbackSpec->GetSpecifier());
    }

    Path path = primPath;
    while (!path.IsEmpty() && !path.IsAbsoluteRoot()) {
        const PrimIndex* idx =
            (path == primPath) ? primIndex : graph->GetPrimIndex(path);
        const Spec* specFallback = (path == primPath) ? fallbackSpec : nullptr;
        Specifier spec = GetComposedSpecifier(graph, idx, path, specFallback);
        if (!IsDefiningSpecifier(spec)) return false;
        path = path.GetParentPath();
    }
    return true;
}

bool PrimOrAncestorIsAbstract(const CompositionGraph* graph,
                              const PrimIndex* primIndex,
                              const Path& primPath,
                              const Spec* fallbackSpec) {
    if (!graph) {
        return fallbackSpec &&
               fallbackSpec->GetSpecifier() == Specifier::Class;
    }

    Path path = primPath;
    while (!path.IsEmpty() && !path.IsAbsoluteRoot()) {
        const PrimIndex* idx =
            (path == primPath) ? primIndex : graph->GetPrimIndex(path);
        const Spec* specFallback = (path == primPath) ? fallbackSpec : nullptr;
        if (GetComposedSpecifier(graph, idx, path, specFallback) ==
            Specifier::Class) {
            return true;
        }
        path = path.GetParentPath();
    }
    return false;
}

Token GetComposedKind(const CompositionGraph* graph,
                      const PrimIndex* primIndex,
                      const Path& primPath,
                      const Spec* fallbackSpec) {
    if (graph && primIndex) {
        auto* val = GetFirstPrimField(graph, primIndex, primPath,
                                      FieldNames::kind);
        if (val) {
            if (auto* t = val->Get<Token>()) return *t;
        }
        return Token("");
    }
    return fallbackSpec ? fallbackSpec->GetKind() : Token("");
}

bool IsModelKind(const Token& kind) {
    return kind == "component" || kind == "assembly" || kind == "group";
}

bool IsGroupModelKind(const Token& kind) {
    return kind == "assembly" || kind == "group";
}

bool PrimIsInModelHierarchy(const CompositionGraph* graph,
                            const PrimIndex* primIndex,
                            const Path& primPath,
                            const Spec* fallbackSpec) {
    if (!IsModelKind(GetComposedKind(graph, primIndex, primPath, fallbackSpec)))
        return false;
    if (!graph) return true;

    Path path = primPath.GetParentPath();
    while (!path.IsEmpty() && !path.IsAbsoluteRoot()) {
        const PrimIndex* idx = graph->GetPrimIndex(path);
        if (!IsGroupModelKind(GetComposedKind(graph, idx, path, nullptr)))
            return false;
        path = path.GetParentPath();
    }
    return true;
}

const Dictionary* GetFallbackPrimTypesDictionary(const CompositionGraph* graph) {
    if (!graph) return nullptr;
    if (const auto* field =
            graph->composedLayerSpec.GetField(FieldNames::fallbackPrimTypes)) {
        if (const auto* dict = field->Get<Dictionary>()) return dict;
    }
    if (graph->GetNumLayers() > 0) {
        if (const auto* field = graph->GetRootLayer().GetLayerSpec().GetField(
                FieldNames::fallbackPrimTypes)) {
            if (const auto* dict = field->Get<Dictionary>()) return dict;
        }
    }
    return nullptr;
}

std::string ResolvePrimSchemaTypeName(const Token& authoredTypeName,
                                      const CompositionGraph* graph) {
    return SchemaRegistry::GetInstance().ResolvePrimTypeName(
        authoredTypeName.GetString(), GetFallbackPrimTypesDictionary(graph));
}

void AppendUniqueToken(std::vector<Token>& values, const Token& value) {
    if (std::find(values.begin(), values.end(), value) == values.end())
        values.push_back(value);
}

void AppendUniqueString(std::vector<std::string>& values,
                        const std::string& value) {
    if (std::find(values.begin(), values.end(), value) == values.end())
        values.push_back(value);
}

std::vector<std::string> ToStringVector(const std::vector<Token>& tokens) {
    std::vector<std::string> result;
    result.reserve(tokens.size());
    for (const auto& token : tokens) result.push_back(token.GetString());
    return result;
}

std::vector<Token> GetAuthoredAppliedSchemasForPrim(
        const CompositionGraph* graph,
        const PrimIndex* primIndex,
        const Path& primPath,
        const Spec* fallbackSpec) {
    if (graph && primIndex) {
        auto combined = ResolveListOpField<Token>(
            *graph, primIndex, primPath, FieldNames::apiSchemas,
            SpecType::Prim);
        return combined ? combined->GetItems() : std::vector<Token>{};
    }
    if (!fallbackSpec) return {};
    const auto* field = fallbackSpec->GetField(FieldNames::apiSchemas);
    if (!field || field->IsBlock()) return {};
    if (const auto* listOp = field->Get<ListOp<Token>>()) {
        return listOp->GetItems();
    }
    return {};
}

std::vector<std::string> GetPrimDefinitionAPISchemas(
        const std::vector<Token>& authoredAPISchemas,
        const std::string& schemaTypeName) {
    std::vector<std::string> result = ToStringVector(authoredAPISchemas);
    if (schemaTypeName.empty()) return result;

    auto autoApplied =
        SchemaRegistry::GetInstance().GetAutoAppliedAPISchemasForType(
            schemaTypeName);
    for (const auto& schema : autoApplied) AppendUniqueString(result, schema);
    return result;
}

std::vector<Token> GetComposedAppliedSchemasForPrim(
        const std::vector<Token>& authoredAPISchemas,
        const std::string& schemaTypeName) {
    std::vector<Token> result = authoredAPISchemas;
    if (schemaTypeName.empty()) return result;

    auto& registry = SchemaRegistry::GetInstance();
    for (const auto& schema :
         registry.GetBuiltInAPISchemasForType(schemaTypeName)) {
        AppendUniqueToken(result, Token(schema));
    }
    for (const auto& schema :
         registry.GetAutoAppliedAPISchemasForType(schemaTypeName)) {
        AppendUniqueToken(result, Token(schema));
    }
    return result;
}

} // anonymous namespace

// ============================================================
// UsdPrim
// ============================================================

Specifier UsdPrim::GetSpecifier() const {
    return GetComposedSpecifier(graph_, primIndex_, path_, spec_);
}

Token UsdPrim::GetTypeName() const {
    if (graph_ && primIndex_) {
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePrimPath(entry, path_);
            if (sourcePath.IsEmpty()) continue;
            const auto* spec = layer.GetPrimSpec(sourcePath);
            if (!spec) continue;
            Token tn = spec->GetTypeName();
            if (!tn.IsEmpty()) return tn;
        }
        return Token("");
    }
    return spec_ ? spec_->GetTypeName() : Token("");
}

bool UsdPrim::IsActive() const {
    if (graph_ && primIndex_)
        return PrimAndAncestorsAreActive(graph_, primIndex_, path_, spec_);
    return spec_ ? spec_->GetActive() : true;
}

bool UsdPrim::IsLoaded() const {
    // Payloads are composed eagerly; any valid populated prim is loaded.
    return IsValid();
}

bool UsdPrim::IsModel() const {
    return PrimIsInModelHierarchy(graph_, primIndex_, path_, spec_);
}

bool UsdPrim::IsInstanceable() const {
    if (graph_ && primIndex_) {
        auto* val = GetFirstPrimField(graph_, primIndex_, path_, FieldNames::instanceable);
        if (val) {
            if (auto* b = val->Get<bool>()) return *b;
        }
        return false;
    }
    return spec_ ? spec_->GetInstanceable() : false;
}

bool UsdPrim::IsInstance() const {
    if (!stage_) return false;
    return stage_->IsInstance(path_);
}

bool UsdPrim::IsPrototype() const {
    if (!stage_) return false;
    return stage_->IsPrototypeRoot(path_);
}

bool UsdPrim::IsInPrototype() const {
    if (!stage_) return false;
    // Walk ancestors checking if any is a prototype root
    Path p = path_;
    while (!p.IsEmpty() && !p.IsAbsoluteRoot()) {
        if (stage_->IsPrototypeRoot(p)) return true;
        p = p.GetParentPath();
    }
    return false;
}

UsdPrim UsdPrim::GetPrototype() const {
    if (!stage_) return {};
    Path protoPath = stage_->GetPrototypePath(path_);
    if (protoPath.IsEmpty()) return {};
    return stage_->GetPrimAtPath(protoPath);
}

std::vector<UsdPrim> UsdPrim::GetInstances() const {
    if (!stage_) return {};
    const auto& instancePaths = stage_->GetInstancePaths(path_);
    std::vector<UsdPrim> result;
    result.reserve(instancePaths.size());
    for (const auto& ip : instancePaths) {
        auto prim = stage_->GetPrimAtPath(ip);
        if (prim.IsValid()) result.push_back(std::move(prim));
    }
    return result;
}

bool UsdPrim::IsHidden() const {
    if (graph_ && primIndex_) {
        auto* val = GetFirstPrimField(graph_, primIndex_, path_, FieldNames::hidden);
        if (val) {
            if (auto* b = val->Get<bool>()) return *b;
        }
        return false;
    }
    return spec_ ? spec_->GetHidden() : false;
}

Token UsdPrim::GetKind() const {
    return GetComposedKind(graph_, primIndex_, path_, spec_);
}

String UsdPrim::GetDocumentation() const {
    if (graph_ && primIndex_) {
        auto* val = GetFirstPrimField(graph_, primIndex_, path_, FieldNames::documentation);
        if (val) {
            if (auto* s = val->Get<String>()) return *s;
        }
        return "";
    }
    return spec_ ? spec_->GetDocumentation() : "";
}

std::optional<Value> UsdPrim::GetPrimMetadata(const Token& fieldName) const {
    if (graph_ && primIndex_) {
        // Spec §12.2.5: Dictionary-typed fields combine across the opinion
        // stack per §6.6.2.1 rather than taking the strongest-authored
        // value. Asset-path resolution is folded into the combine per
        // opinion, anchored to that opinion's layer.
        const auto* desc = FieldRegistry::Instance().Find(fieldName);
        if (desc && desc->type == TypeId::Dictionary) {
            return ResolveDictionaryField(graph_, primIndex_, path_,
                                          fieldName, graph_->resolver);
        }
        // Spec §12.2.6: ListOp-typed metadata fields combine across the
        // opinion stack per §6.6.3.6. Dispatch on the actual stored
        // listop element type. The order of attempts below is the
        // extension point for adding more element types.
        //   ListOp<Path>         — connectionPaths, targetPaths
        //   ListOp<Token>        — apiSchemas (spec §13.2.1.2)
        //   ListOp<std::string>  — clipSets (§13.2.1.2.2), variantSetNames
        if (desc && desc->isListOp) {
            if (auto p = ResolveListOpField<Path>(*graph_, primIndex_, path_,
                                                   fieldName, SpecType::Prim)) {
                return Value(std::move(*p));
            }
            if (auto t = ResolveListOpField<Token>(*graph_, primIndex_, path_,
                                                    fieldName, SpecType::Prim)) {
                return Value(std::move(*t));
            }
            if (auto s = ResolveListOpField<std::string>(*graph_, primIndex_,
                                                         path_, fieldName,
                                                         SpecType::Prim)) {
                return Value(std::move(*s));
            }
            return std::nullopt;
        }
        auto result = GetFirstPrimFieldWithProvenance(graph_, primIndex_, path_, fieldName);
        if (!result.value) return std::nullopt;
        // If the field may contain asset paths, resolve them (spec §5)
        if (graph_->resolver) {
            auto typeId = result.value->GetTypeId();
            if (typeId == TypeId::Asset || typeId == TypeId::Dictionary) {
                Value copy = *result.value;
                ResolveAssetValueOrDict(copy, graph_->layerPaths[result.layerIndex],
                                        graph_->resolver);
                return copy;
            }
        }
        return *result.value;
    }
    if (spec_) {
        auto* val = spec_->GetField(fieldName);
        if (val) return *val;
    }
    return std::nullopt;
}

bool UsdPrim::IsDefined() const {
    // Stage Population: a prim is defined when its resolved specifier and
    // every ancestor's resolved specifier are defining (def or class).
    return PrimAndAncestorsAreDefined(graph_, primIndex_, path_, spec_);
}

bool UsdPrim::IsAbstract() const {
    // Stage Population: a prim is abstract when it or any ancestor resolves
    // to the class specifier.
    return PrimOrAncestorIsAbstract(graph_, primIndex_, path_, spec_);
}

std::vector<UsdPrim> UsdPrim::GetChildren() const {
    if (layerIndex_ == kNoLayer && !graph_) return {};

    // If this is an instance, return the prototype's children instead
    if (stage_ && stage_->IsInstance(path_)) {
        Path protoPath = stage_->GetPrototypePath(path_);
        if (!protoPath.IsEmpty()) {
            auto protoPrim = stage_->GetPrimAtPath(protoPath);
            if (protoPrim.IsValid()) return protoPrim.GetChildren();
        }
        return {};
    }

    // Collect child prim names from the graph or single layer
    std::unordered_set<Token, Token::Hash> childNameSet;
    std::vector<Token> childNames;

    if (graph_) {
        const auto& indexedNames = graph_->GetComposedChildNames(path_);
        for (const auto& name : indexedNames) {
            if (childNameSet.insert(name).second)
                childNames.push_back(name);
        }
    } else {
        const auto& indexedNames = GetLayer().GetChildNames(path_);
        for (const auto& name : indexedNames) {
            if (childNameSet.insert(name).second)
                childNames.push_back(name);
        }
    }

    SortAndApplyComposedPrimOrder(
        childNames, graph_, primIndex_, path_, spec_);

    // Build UsdPrim objects, filtering by population rules
    std::vector<UsdPrim> result;
    for (const auto& childName : childNames) {
        Path childPath = path_.AppendChild(childName);
        if (stage_ && !stage_->PopulationMaskIncludes(childPath))
            continue;

        if (graph_) {
            const PrimIndex* childIdx = graph_->GetPrimIndex(childPath);
            if (!childIdx || childIdx->entries.empty()) continue;

            // Get the strongest spec for population check
            const Spec* strongestSpec =
                GetStrongestPrimSpec(graph_, childIdx, childPath);
            if (!strongestSpec) continue;

            // Check composed specifier
            Specifier spec = GetComposedSpecifier(graph_, childIdx, childPath, strongestSpec);
            if (spec == Specifier::Over) continue;

            // Check active
            auto* activeVal =
                GetFirstPrimField(graph_, childIdx, childPath,
                                  FieldNames::active);
            if (!IsActiveValueEnabled(activeVal)) continue;

            result.emplace_back(childPath, childName, strongestSpec,
                               graph_, childIdx, layerIndex_, stage_);
        } else {
            const auto* childSpec = GetLayer().GetPrimSpec(childPath);
            if (!childSpec) continue;
            if (childSpec->GetSpecifier() == Specifier::Over) continue;
            if (!childSpec->GetActive()) continue;
            result.emplace_back(childPath, childName, childSpec, layerIndex_);
        }
    }

    return result;
}

UsdPrim UsdPrim::GetChild(const Token& name) const {
    if (layerIndex_ == kNoLayer && !graph_) return {};

    Path childPath = path_.AppendChild(name);
    if (stage_ && !stage_->PopulationMaskIncludes(childPath)) return {};

    if (graph_) {
        const PrimIndex* childIdx = graph_->GetPrimIndex(childPath);
        if (!childIdx || childIdx->entries.empty()) return {};

        const Spec* strongestSpec =
            GetStrongestPrimSpec(graph_, childIdx, childPath);
        if (!strongestSpec) return {};

        Specifier spec = GetComposedSpecifier(graph_, childIdx, childPath, strongestSpec);
        if (spec == Specifier::Over) return {};

        auto* activeVal =
            GetFirstPrimField(graph_, childIdx, childPath,
                              FieldNames::active);
        if (!IsActiveValueEnabled(activeVal)) return {};

        return UsdPrim(childPath, name, strongestSpec, graph_, childIdx, layerIndex_, stage_);
    }

    const auto* childSpec = GetLayer().GetPrimSpec(childPath);
    if (!childSpec) return {};
    if (childSpec->GetSpecifier() == Specifier::Over) return {};
    if (!childSpec->GetActive()) return {};
    return UsdPrim(childPath, name, childSpec, layerIndex_);
}

bool UsdPrim::HasChild(const Token& name) const {
    return GetChild(name).IsValid();
}

std::vector<Token> UsdPrim::GetPropertyNames() const {
    if (layerIndex_ == kNoLayer && !graph_) return {};

    std::unordered_set<Token, Token::Hash> nameSet;
    std::vector<Token> names;
    names.reserve(16);

    if (graph_ && primIndex_) {
        // Use Layer property index for O(1) lookup per layer
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePrimPath = OpinionSourcePrimPath(entry, path_);
            if (sourcePrimPath.IsEmpty()) continue;

            const auto& layerProps = layer.GetPropertyNames(sourcePrimPath);
            for (const auto& propName : layerProps) {
                if (nameSet.insert(propName).second) {
                    names.push_back(propName);
                }
            }
        }
    } else if (layerIndex_ != kNoLayer) {
        const auto& layerProps = GetLayer().GetPropertyNames(path_);
        for (const auto& propName : layerProps) {
            if (nameSet.insert(propName).second) {
                names.push_back(propName);
            }
        }
    }

    // Add schema-defined properties not already authored
    auto primDef = GetPrimDefinition();
    for (const auto& [propName, propDef] : primDef.properties) {
        Token tok(propName);
        if (nameSet.insert(tok).second) {
            names.push_back(tok);
        }
    }

    SortAndApplyPropertyOrder(names, graph_, primIndex_, path_, spec_);

    return names;
}

UsdAttribute UsdPrim::GetAttribute(const Token& name) const {
    if (layerIndex_ == kNoLayer && !graph_) return {};

    Path propPath = path_.AppendProperty(name);

    // Look up schema property definition
    auto primDef = GetPrimDefinition();
    auto* schemaProp = primDef.GetPropertyDef(name);

    if (graph_ && primIndex_) {
        // Walk opinion stack for the authored spec
        const Spec* authoredSpec = nullptr;
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePath(entry, propPath);
            if (sourcePath.IsEmpty()) continue;
            authoredSpec = layer.GetAttributeSpec(sourcePath);
            if (authoredSpec) break;
        }

        if (authoredSpec && schemaProp) {
            return UsdAttribute(propPath, name, authoredSpec, *schemaProp,
                               graph_, primIndex_, layerIndex_);
        }
        if (authoredSpec) {
            return UsdAttribute(propPath, name, authoredSpec,
                               graph_, primIndex_, layerIndex_);
        }
        if (schemaProp && schemaProp->typeName != "rel") {
            return UsdAttribute(propPath, name, *schemaProp,
                               graph_, primIndex_, layerIndex_);
        }
        return {};
    }

    // Legacy single-layer path
    if (layerIndex_ == kNoLayer) return {};
    const auto* spec = GetLayer().GetAttributeSpec(propPath);
    if (spec && schemaProp) {
        return UsdAttribute(propPath, name, spec, *schemaProp, layerIndex_);
    }
    if (spec) {
        return UsdAttribute(propPath, name, spec, layerIndex_);
    }
    if (schemaProp && schemaProp->typeName != "rel") {
        return UsdAttribute(propPath, name, *schemaProp, layerIndex_);
    }
    return {};
}

UsdRelationship UsdPrim::GetRelationship(const Token& name) const {
    if (layerIndex_ == kNoLayer && !graph_) return {};

    Path propPath = path_.AppendProperty(name);

    // Check schema for relationship definition
    auto primDef = GetPrimDefinition();
    auto* schemaProp = primDef.GetPropertyDef(name);
    bool isSchemaRel = schemaProp && schemaProp->typeName == "rel";

    if (graph_ && primIndex_) {
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePath(entry, propPath);
            if (sourcePath.IsEmpty()) continue;
            const auto* spec = layer.GetRelationshipSpec(sourcePath);
            if (spec) {
                return UsdRelationship(propPath, name, spec, graph_, primIndex_, layerIndex_);
            }
        }
        // Schema-defined relationship with no authored spec
        if (isSchemaRel) {
            return UsdRelationship(propPath, name, graph_, primIndex_, layerIndex_);
        }
        return {};
    }

    if (layerIndex_ == kNoLayer) return {};
    const auto* spec = GetLayer().GetRelationshipSpec(propPath);
    if (spec) {
        return UsdRelationship(propPath, name, spec, layerIndex_);
    }
    // Schema-defined relationship with no authored spec
    if (isSchemaRel) {
        return UsdRelationship(propPath, name, layerIndex_);
    }
    return {};
}

void UsdPrim::AppendAttributeNames(std::vector<Token>& names) const {
    if (layerIndex_ == kNoLayer && !graph_) return;

    bool timing = AttrTimingEnabled();
    auto tStart = timing ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point{};

    auto appendUnique = [&names](const Token& name) {
        if (std::find(names.begin(), names.end(), name) != names.end())
            return;
        names.push_back(name);
    };
    auto appendUniqueRange = [&names, &appendUnique](const std::vector<Token>& src) {
        if (src.empty()) return;
        names.reserve(names.size() + src.size());
        if (names.empty()) {
            names.insert(names.end(), src.begin(), src.end());
            return;
        }
        for (const auto& name : src) appendUnique(name);
    };

    // Fold typeName discovery and apiSchemas detection into the same opinion
    // walk we need for property names — avoids two separate GetTypeName /
    // GetAppliedSchemas passes and the vector<string> allocations they'd do.
    Token typeName;
    bool hasApiSchemas = false;

    if (graph_ && primIndex_) {
        for (const auto& entry : primIndex_->entries) {
            if (timing) g_attrStats.entries_visited++;
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePrimPath = entry.pathMapping->isIdentity ? path_ :
                OpinionSourcePrimPath(entry, path_);
            if (sourcePrimPath.IsEmpty()) continue;

            if (typeName.IsEmpty() || !hasApiSchemas) {
                if (const auto* primSpec = layer.GetPrimSpec(sourcePrimPath)) {
                    if (typeName.IsEmpty()) typeName = primSpec->GetTypeName();
                    if (!hasApiSchemas && primSpec->GetField(FieldNames::apiSchemas))
                        hasApiSchemas = true;
                }
            }

            const auto& layerAttrs = layer.GetAttributeNames(sourcePrimPath);
            appendUniqueRange(layerAttrs);
        }
    } else if (layerIndex_ != kNoLayer) {
        if (timing) g_attrStats.entries_visited++;
        if (const auto* primSpec = GetLayer().GetPrimSpec(path_)) {
            typeName = primSpec->GetTypeName();
            hasApiSchemas = primSpec->GetField(FieldNames::apiSchemas) != nullptr;
        }
        const auto& layerAttrs = GetLayer().GetAttributeNames(path_);
        appendUniqueRange(layerAttrs);
    }

    auto tSchemaStart = timing ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
    // Only touch the schema registry when the prim could actually inherit
    // schema-defined attributes. Avoid GetPrimDefinition() here: it returns
    // PrimDefinition by value, which deep-copies the whole properties map
    // on every call. Instead, hit GetOrBuildPrimDef directly and read from
    // the cached reference.
    std::string schemaTypeName = ResolvePrimSchemaTypeName(typeName, graph_);
    if (!schemaTypeName.empty() || hasApiSchemas) {
        auto authoredSchemas = hasApiSchemas
            ? GetAuthoredAppliedSchemasForPrim(graph_, primIndex_, path_, spec_)
            : std::vector<Token>{};
        auto apiSchemaStrings =
            GetPrimDefinitionAPISchemas(authoredSchemas, schemaTypeName);
        if (graph_) {
            const PrimDefinition& primDef = graph_->GetOrBuildPrimDef(
                /*primPath unused*/ std::string(),
                schemaTypeName, apiSchemaStrings);
            appendUniqueRange(primDef.attributeNameTokens);
        } else {
            // No-graph fallback: uncached build.
            auto primDef = SchemaRegistry::GetInstance().BuildFullPrimDefinition(
                schemaTypeName, apiSchemaStrings);
            appendUniqueRange(primDef.attributeNameTokens);
        }
    }

    SortAndApplyPropertyOrder(names, graph_, primIndex_, path_, spec_);

    if (timing) {
        auto tEnd = std::chrono::steady_clock::now();
        auto ns = [](auto a, auto b) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
        };
        g_attrStats.ns_entry_walk += ns(tStart, tSchemaStart);
        g_attrStats.ns_schema += ns(tSchemaStart, tEnd);
        g_attrStats.ns_total += ns(tStart, tEnd);
        g_attrStats.unique_names += static_cast<long long>(names.size());
        g_attrStats.calls++;
    }
}

std::vector<Token> UsdPrim::GetAttributeNames() const {
    std::vector<Token> names;
    names.reserve(16);
    AppendAttributeNames(names);
    return names;
}

std::vector<Token> UsdPrim::GetAuthoredAttributeNames() const {
    if (layerIndex_ == kNoLayer && !graph_) return {};

    std::unordered_set<Token, Token::Hash> seen;
    std::vector<Token> names;
    names.reserve(16);

    auto appendUniqueRange = [&names, &seen](const std::vector<Token>& src) {
        names.reserve(names.size() + src.size());
        for (const auto& name : src) {
            if (seen.insert(name).second) {
                names.push_back(name);
            }
        }
    };

    if (graph_ && primIndex_) {
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePrimPath = entry.pathMapping->isIdentity ? path_ :
                OpinionSourcePrimPath(entry, path_);
            if (sourcePrimPath.IsEmpty()) continue;
            appendUniqueRange(layer.GetAttributeNames(sourcePrimPath));
        }
    } else if (layerIndex_ != kNoLayer) {
        appendUniqueRange(GetLayer().GetAttributeNames(path_));
    }

    return names;
}

namespace detail {

size_t GetPrimAttributeCount(const UsdPrim& prim) {
    thread_local std::vector<Token> names;
    names.clear();
    names.reserve(16);
    prim.AppendAttributeNames(names);
    return names.size();
}

} // namespace detail

bool UsdPrim::HasAttribute(const Token& name) const {
    if (layerIndex_ == kNoLayer && !graph_) return false;
    Path propPath = path_.AppendProperty(name);

    if (graph_ && primIndex_) {
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePath(entry, propPath);
            if (sourcePath.IsEmpty()) continue;
            if (layer.GetAttributeSpec(sourcePath)) return true;
        }
    } else if (layerIndex_ != kNoLayer) {
        if (GetLayer().GetAttributeSpec(propPath)) return true;
    }

    auto primDef = GetPrimDefinition();
    auto* schemaProp = primDef.GetPropertyDef(name);
    return schemaProp && schemaProp->typeName != "rel";
}

bool UsdPrim::HasRelationship(const Token& name) const {
    return GetRelationship(name).IsValid();
}

std::vector<Token> UsdPrim::GetPropertyOrder() const {
    auto* orderField = GetStrongestPropertyOrderField(
        graph_, primIndex_, path_, spec_);
    if (!orderField) return {};
    if (auto* orderVec = orderField->Get<std::vector<std::string>>()) {
        std::vector<Token> result;
        result.reserve(orderVec->size());
        for (const auto& s : *orderVec) result.emplace_back(s);
        return result;
    }
    return {};
}

// ============================================================
// UsdPrim — Schema queries
// ============================================================

bool UsdPrim::IsA(const Token& typeName) const {
    std::string schemaTypeName = ResolvePrimSchemaTypeName(
        GetTypeName(), graph_);
    if (schemaTypeName.empty()) return false;
    return SchemaRegistry::GetInstance().IsA(
        schemaTypeName, typeName.GetString());
}

bool UsdPrim::HasAPI(const Token& schemaName) const {
    auto schemas = GetAppliedSchemas();
    for (const auto& s : schemas) {
        if (s == schemaName) return true;
        const std::string& str = s.GetString();
        auto colonPos = str.find(':');
        if (colonPos != std::string::npos && str.substr(0, colonPos) == schemaName.GetString())
            return true;
    }
    return false;
}

bool UsdPrim::HasAPI(const Token& schemaName, const Token& instanceName) const {
    auto schemas = GetAppliedSchemas();
    Token target(schemaName.GetString() + ":" + instanceName.GetString());
    for (const auto& s : schemas) {
        if (s == target) return true;
    }
    return false;
}

std::vector<Token> UsdPrim::GetAppliedSchemas() const {
    auto authored = GetAuthoredAppliedSchemasForPrim(
        graph_, primIndex_, path_, spec_);
    return GetComposedAppliedSchemasForPrim(
        authored, ResolvePrimSchemaTypeName(GetTypeName(), graph_));
}

Token UsdPrim::ComputeColorSpaceName() const {
    return ComputePrimColorSpaceName(graph_, primIndex_, path_);
}

// --- Variants (spec §11.2) ---

std::vector<Token> UsdPrim::GetVariantSetNames() const {
    auto toTokens = [](const std::vector<std::string>& strs) {
        std::vector<Token> result;
        result.reserve(strs.size());
        for (const auto& s : strs) result.emplace_back(s);
        return result;
    };

    if (graph_ && primIndex_) {
        std::optional<ListOp<std::string>> combined;
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePrimPath(entry, path_);
            if (sourcePath.IsEmpty()) continue;
            const auto* spec = layer.GetPrimSpec(sourcePath);
            if (!spec) continue;
            auto* field = spec->GetField(FieldNames::variantSetNames);
            if (!field) continue;
            auto* listOp = field->Get<ListOp<std::string>>();
            if (!listOp) continue;
            if (!combined) combined = *listOp;
            else combined = combined->Combine(*listOp);
        }
        if (combined) return toTokens(combined->GetItems());
        return {};
    }
    if (!spec_) return {};
    auto* field = spec_->GetField(FieldNames::variantSetNames);
    if (!field) return {};
    if (auto* listOp = field->Get<ListOp<std::string>>()) {
        return toTokens(listOp->GetItems());
    }
    return {};
}

bool UsdPrim::HasVariantSet(const Token& setName) const {
    auto names = GetVariantSetNames();
    for (const auto& n : names) {
        if (n == setName) return true;
    }
    return false;
}

std::vector<Token> UsdPrim::GetVariantNames(const Token& setName) const {
    // Walks Variant specs at /primPath{setName=*} across opinion entries.
    // A given (setName, variantName) may be authored on multiple layers;
    // dedup by insertion into a small ordered list so first-seen wins.
    auto collectFromLayer = [&](const Layer& layer, const Path& sourcePath,
                                 std::vector<Token>& out) {
        layer.ForEachSpec([&](const Path& specPath, const Spec& spec) {
            if (spec.GetType() != SpecType::Variant) return;
            // Variant paths carry variantSelections on their last element.
            // Match: same parent + same prim name + a selection for setName.
            if (specPath.GetName() != sourcePath.GetName()) return;
            if (specPath.GetParentPath() != sourcePath.GetParentPath()) return;
            const auto& elems = GetCachedPathElements(specPath);
            if (elems.empty()) return;
            for (const auto& vs : elems.back().variantSelections) {
                if (vs.setName != setName) continue;
                if (vs.variantName.IsEmpty()) continue;
                for (const auto& existing : out) {
                    if (existing == vs.variantName) return;
                }
                out.push_back(vs.variantName);
                return;
            }
        });
    };

    std::vector<Token> names;
    if (graph_ && primIndex_) {
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePrimPath(entry, path_);
            if (sourcePath.IsEmpty()) continue;
            collectFromLayer(layer, sourcePath, names);
        }
        return names;
    }
    return names;
}

Token UsdPrim::GetVariantSelection(const Token& setName) const {
    // Strongest-wins walk across the opinion stack. Matches the composition
    // logic in ResolveVariantSelections so this query agrees with whatever
    // the composed stage actually selected.
    if (graph_ && primIndex_) {
        for (const auto& entry : primIndex_->entries) {
            const Layer& layer = graph_->GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePrimPath(entry, path_);
            if (sourcePath.IsEmpty()) continue;
            const auto* spec = layer.GetPrimSpec(sourcePath);
            if (!spec) continue;
            const auto* field = spec->GetField(FieldNames::variantSelection);
            if (!field) continue;
            const auto* dict = field->Get<Dictionary>();
            if (!dict) continue;
            auto it = dict->find(setName.GetString());
            if (it == dict->end()) continue;
            if (const auto* tok = it->second.Get<Token>()) {
                if (!tok->IsEmpty()) return *tok;
            } else if (const auto* s = it->second.Get<String>()) {
                if (!s->empty()) return Token(*s);
            }
        }
        return {};
    }
    if (!spec_) return {};
    const auto* field = spec_->GetField(FieldNames::variantSelection);
    if (!field) return {};
    const auto* dict = field->Get<Dictionary>();
    if (!dict) return {};
    auto it = dict->find(setName.GetString());
    if (it == dict->end()) return {};
    if (const auto* tok = it->second.Get<Token>()) {
        return *tok;
    } else if (const auto* s = it->second.Get<String>()) {
        return Token(*s);
    }
    return {};
}

bool UsdPrim::SetVariantSelection(const Token& setName,
                                   const Token& variantName,
                                   size_t layerIndex) {
    if (!graph_) return false;
    if (layerIndex >= graph_->GetNumLayers()) return false;

    Layer& layer = graph_->GetMutableLayer(layerIndex);
    Spec* spec = layer.GetMutablePrimSpec(path_);
    if (!spec) {
        // No spec on this layer yet — create an over.
        Spec over(SpecType::Prim);
        over.SetSpecifier(Specifier::Over);
        layer.SetSpec(path_, std::move(over));
        spec = layer.GetMutablePrimSpec(path_);
        if (!spec) return false;
    }

    Dictionary dict;
    if (auto* field = spec->GetField(FieldNames::variantSelection)) {
        if (const auto* existing = field->Get<Dictionary>()) {
            dict = *existing;
        }
    }
    // Store as Token so ResolveVariantSelections picks it up directly.
    // Empty variantName clears the selection per USD conventions.
    if (variantName.IsEmpty()) {
        dict.erase(setName.GetString());
    } else {
        dict[setName.GetString()] = Value(variantName);
    }
    spec->SetField(FieldNames::variantSelection, Value(std::move(dict)));
    return true;
}

PrimDefinition UsdPrim::GetPrimDefinition() const {
    std::string schemaTypeName = ResolvePrimSchemaTypeName(
        GetTypeName(), graph_);
    auto authoredSchemas = GetAuthoredAppliedSchemasForPrim(
        graph_, primIndex_, path_, spec_);
    auto apiSchemas =
        GetPrimDefinitionAPISchemas(authoredSchemas, schemaTypeName);
    if (graph_) {
        return graph_->GetOrBuildPrimDef(
            path_.GetText(), schemaTypeName, apiSchemas);
    }
    // Single-layer mode (no graph): uncached fallback
    return SchemaRegistry::GetInstance().BuildFullPrimDefinition(
        schemaTypeName, apiSchemas);
}

// ============================================================
// Stage
// ============================================================

Stage Stage::Open(const std::string& filePath, AssetResolver resolver) {
    return OpenImpl(filePath, std::move(resolver), {}, false);
}

Stage Stage::OpenMasked(const std::string& filePath,
                        const std::vector<Path>& populationMask,
                        AssetResolver resolver) {
    return OpenImpl(filePath, std::move(resolver),
                    std::vector<Path>(populationMask.begin(),
                                      populationMask.end()),
                    true);
}

Stage Stage::OpenImpl(const std::string& filePath,
                      AssetResolver resolver,
                      std::vector<Path> populationMask,
                      bool hasPopulationMask) {
    Stage stage;
    const bool explicitEmptyPopulationMask =
        hasPopulationMask && populationMask.empty();
    stage.hasPopulationMask_ = hasPopulationMask && !explicitEmptyPopulationMask;
    if (hasPopulationMask) {
        stage.populationMask_ = NormalizePopulationMask(
            std::move(populationMask));
    }

    if (!resolver) resolver = DefaultResolve;

    PhaseTimer timer;

    std::string resolvedPath = resolver("", filePath);
    if (resolvedPath.empty()) resolvedPath = filePath;

    auto parseResult = ParseUsdFile(resolvedPath);
    timer.lap("parse root layer");
    if (!parseResult.success) {
        stage.error_ = "Failed to parse " + resolvedPath + ": " + parseResult.error;
        return stage;
    }

    auto composeResult = Compose(parseResult.layer, resolvedPath, std::move(resolver));
    timer.lap("compose (total)");
    if (!composeResult.success) {
        stage.error_ = "Failed to compose: " + composeResult.error;
        stage.diagnostics_ = std::move(composeResult.diagnostics);
        return stage;
    }

    stage.graph_ = std::move(composeResult.graph);
    stage.diagnostics_ = std::move(composeResult.diagnostics);
    stage.valid_ = true;
    stage.Populate();
    timer.lap("populate");
    return stage;
}

Stage Stage::CreateInMemory() {
    Stage stage;
    stage.graph_.layers.push_back(std::make_shared<Layer>());  // empty root layer
    stage.graph_.layerPaths.push_back("");
    stage.graph_.layerRetimings.push_back({0.0, 1.0});
    stage.valid_ = true;
    return stage;
}

UsdPrim Stage::DefinePrim(const Path& path, const Token& typeName) {
    if (!valid_ || path.IsEmpty()) return {};

    Layer& rootLayer = graph_.GetRootLayer();
    const auto& elems = GetCachedPathElements(path);
    const int depth = static_cast<int>(elems.size());

    // Helper: ensure a spec + prim index entry exists for a path.
    auto ensureSpec = [&](const Path& p, bool isTarget) {
        if (!rootLayer.GetPrimSpec(p)) {
            Spec primSpec(SpecType::Prim);
            primSpec.SetSpecifier(Specifier::Def);
            if (isTarget && !typeName.IsEmpty()) {
                primSpec.SetTypeName(typeName);
            }
            rootLayer.SetSpec(p, std::move(primSpec));
        } else if (isTarget) {
            auto* spec = rootLayer.GetPrimSpec(p);
            const_cast<Spec*>(spec)->SetSpecifier(Specifier::Def);
            if (!typeName.IsEmpty()) {
                const_cast<Spec*>(spec)->SetTypeName(typeName);
            }
        }
        if (!graph_.HasPrimIndex(p)) {
            OpinionEntry entry;
            entry.layerIndex = 0;
            // pathMapping defaults to the shared identity singleton.
            entry.sourcePath = p;
            graph_.EnsurePrimIndex(p).entries.push_back(std::move(entry));
            graph_.AddToChildIndex(p);
        }
    };

    // Fast path: if the parent already has a prim index entry, only create
    // the target.  Check the prim index rather than constructing a parent
    // Path object (which would copy O(depth) elements).
    // For depth==1 the parent is absolute root, which always "exists".
    bool parentExists = (depth <= 1);
    if (!parentExists) {
        // Check if any existing prim index key matches our parent prefix.
        // GetParentPath is O(depth) from the element copy, but we only
        // call it once on the fast path.
        parentExists = graph_.HasPrimIndex(path.GetParentPath());
    }
    if (parentExists) {
        ensureSpec(path, true);
    } else {
        // Slow path: walk ancestors top-down, creating any that are missing.
        // Build paths incrementally via AppendChild.
        Path current = Path::AbsoluteRoot();
        for (int k = 0; k < depth; ++k) {
            current = current.AppendChild(elems[k].name);
            ensureSpec(current, k == depth - 1);
        }
    }

    // Mark population dirty — defer rebuild until Traverse() is called
    populateDirty_ = true;

    // Return the prim directly — we just created/confirmed it, so skip the
    // ancestor active check that GetPrimAtPath does (O(depth) cost).
    const PrimIndex* idx = graph_.GetPrimIndex(path);
    const Spec* spec = rootLayer.GetPrimSpec(path);
    if (!idx || !spec) return {};
    if (!PopulationMaskIncludes(path)) return {};
    return UsdPrim(path, path.GetName(), spec, &graph_, idx, size_t(0));
}

bool Stage::Recompose() {
    if (!valid_) return false;

    // Save the root layer and its path — composition will copy it
    Layer rootLayer = graph_.GetRootLayer();
    std::string rootPath = graph_.layerPaths.empty() ? "" : graph_.layerPaths[0];

    // Carry the installed resolver across recomposition. Without this a
    // custom resolver supplied to Stage::Open would be silently replaced by
    // the default filesystem resolver every time an authoring edit triggers
    // a recompose (variant selection, add_reference, etc.).
    auto result = Compose(rootLayer, rootPath, graph_.resolver);
    diagnostics_ = std::move(result.diagnostics);
    if (!result.success) {
        error_ = result.error;
        return false;
    }

    graph_ = std::move(result.graph);
    Populate();
    return true;
}

Stage Stage::CreateFromComposedLayer(Layer composedLayer) {
    Stage stage;

    // Wrap the single layer in a graph with identity prim indices
    stage.graph_.layers.push_back(std::make_shared<Layer>(std::move(composedLayer)));
    stage.graph_.layerPaths.push_back("");
    stage.graph_.layerRetimings.push_back({0.0, 1.0});
    stage.graph_.composedLayerSpec = stage.graph_.GetLayer(0).GetLayerSpec();
    stage.graph_.arcOriginLayerCount = 1;

    // Build prim indices for the single layer
    const Layer& layer = stage.graph_.GetLayer(0);
    for (const auto& specPath : layer.GetSpecPaths()) {
        const auto* spec = layer.GetSpec(specPath);
        if (!spec || spec->GetType() != SpecType::Prim) continue;

        OpinionEntry entry;
        entry.layerIndex = 0;
        // pathMapping defaults to the shared identity singleton.
        entry.sourcePath = specPath;
        stage.graph_.EnsurePrimIndex(specPath).entries.push_back(std::move(entry));
    }

    stage.graph_.BuildChildIndex();
    stage.valid_ = true;
    stage.Populate();
    return stage;
}

Stage Stage::CreateFromComposedLayer(CompositionGraph graph) {
    Stage stage;
    stage.graph_ = std::move(graph);
    stage.valid_ = true;
    stage.Populate();
    return stage;
}

bool Stage::ShouldPopulate(const Spec& spec) const {
    if (spec.GetSpecifier() == Specifier::Over) return false;
    if (!spec.GetActive()) return false;
    return true;
}

void Stage::EnsurePopulated() const {
    if (!populateDirty_) return;
    const_cast<Stage*>(this)->Populate();
}

// Instancing key: hashable key from non-local PrimIndex entries.
// Two instanceable prims share a prototype iff their keys match.
namespace {

struct InstancingKey {
    struct Entry {
        std::string layerPath;   // layer source file path (stable across duplicate loads)
        Path        sourcePath;  // pool-interned prim path; pointer-compare equality

        bool operator==(const Entry& o) const {
            return layerPath == o.layerPath && sourcePath == o.sourcePath;
        }
    };

    std::vector<Entry> entries;
    bool empty = true;

    bool operator==(const InstancingKey& o) const {
        return entries == o.entries;
    }

    struct Hash {
        size_t operator()(const InstancingKey& k) const noexcept {
            size_t h = 0;
            for (const auto& e : k.entries) {
                h ^= std::hash<std::string>()(e.layerPath) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= Path::Hash{}(e.sourcePath) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            return h;
        }
    };
};

// True if `maybeAncestor` is the same path as, or a namespace ancestor of,
// `desc`. Used to classify OpinionEntry origin for instancing: entries
// introduced by arcs at or below the instance belong to its composition;
// entries from arcs on a strict ancestor reach it as local overrides.
bool IsAncestorOrEqual(const Path& maybeAncestor, const Path& desc) {
    if (maybeAncestor.IsEmpty() || desc.IsEmpty()) return false;
    if (maybeAncestor == desc) return true;
    if (!maybeAncestor.HasVariantSelections() && !desc.HasVariantSelections()) {
        const PathData* ancestor = maybeAncestor.GetData_();
        for (const PathData* cur = desc.GetData_(); cur != nullptr; cur = cur->parent) {
            if (cur == ancestor) return true;
        }
        return false;
    }
    const auto& a = GetCachedPathElements(maybeAncestor);
    const auto& d = GetCachedPathElements(desc);
    if (a.size() > d.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].name != d[i].name ||
            a[i].variantSelections != d[i].variantSelections) {
            return false;
        }
    }
    return true;
}

// Build an instancing key from the entries on a PrimIndex that represent
// composition arcs rooted at the instance itself, not arcs rooted at an
// ancestor. Per spec §11.3, an instance's prototype is determined by its
// composition structure; local overrides (including those authored via an
// ancestor's arc that happens to declare a spec at the instance's path)
// must not differentiate prototypes.
//
// Classification: each OpinionEntry carries the path of the arc that
// introduced it (entry.pathMapping->targetPrimPath). If that target is
// descendant-or-self of the instance, the entry is part of the instance's
// composition; if it's a strict ancestor of the instance, the entry is a
// local override and is excluded from the key. Empty targetPrimPath
// denotes a purely local root-sublayer opinion, also excluded.
//
// Layer identity is the source file path (not layerIndex), so duplicate
// loads of the same file produce identical keys, and the key is COW-safe:
// writes that replace graph.layers[i]'s shared_ptr<Layer> don't affect it.
InstancingKey MakeInstancingKey(const PrimIndex* primIndex, const Path& primPath,
                                const CompositionGraph& graph) {
    InstancingKey key;
    for (const auto& entry : primIndex->entries) {
        if (entry.pathMapping->targetPrimPath.IsEmpty()) continue;
        if (!IsAncestorOrEqual(primPath, entry.pathMapping->targetPrimPath)) continue;
        Path sourcePath = entry.pathMapping->isIdentity ? primPath :
            OpinionSourcePrimPath(entry, primPath);
        const std::string& layerPath = graph.layerPaths[entry.layerIndex];
        key.entries.push_back({layerPath, sourcePath});
        key.empty = false;
    }
    return key;
}

enum class CollectionExpansionRule {
    ExplicitOnly,
    ExpandPrims,
    ExpandPrimsAndProperties,
};

std::string CollectionPropertyName(const Token& instanceName,
                                   std::string_view suffix) {
    std::string name = "collection:";
    name += instanceName.GetString();
    if (!suffix.empty()) {
        name += ":";
        name += suffix;
    }
    return name;
}

std::optional<Token> GetCollectionInstanceFromMarkerPath(const Path& path) {
    if (!path.HasProperty()) return std::nullopt;
    const std::string& propName = path.GetPropertyName().GetString();
    constexpr std::string_view kPrefix = "collection:";
    if (propName.rfind(kPrefix, 0) != 0) return std::nullopt;

    std::string_view instance(propName);
    instance.remove_prefix(kPrefix.size());
    if (instance.empty() || instance.find(':') != std::string_view::npos)
        return std::nullopt;
    return Token(instance);
}

CollectionExpansionRule GetCollectionExpansionRule(
        const UsdPrim& prim,
        const Token& instanceName) {
    auto attr = prim.GetAttribute(
        Token(CollectionPropertyName(instanceName, "expansionRule")));
    if (attr.IsValid()) {
        auto resolved = attr.Get();
        if (resolved.found) {
            if (auto* tok = resolved.value.Get<Token>()) {
                if (*tok == "explicitOnly")
                    return CollectionExpansionRule::ExplicitOnly;
                if (*tok == "expandPrimsAndProperties")
                    return CollectionExpansionRule::ExpandPrimsAndProperties;
                if (*tok == "expandPrims")
                    return CollectionExpansionRule::ExpandPrims;
            }
            if (auto* str = resolved.value.Get<String>()) {
                if (*str == "explicitOnly")
                    return CollectionExpansionRule::ExplicitOnly;
                if (*str == "expandPrimsAndProperties")
                    return CollectionExpansionRule::ExpandPrimsAndProperties;
                if (*str == "expandPrims")
                    return CollectionExpansionRule::ExpandPrims;
            }
        }
    }
    return CollectionExpansionRule::ExpandPrims;
}

bool GetCollectionIncludeRoot(const UsdPrim& prim,
                              const Token& instanceName) {
    auto attr = prim.GetAttribute(
        Token(CollectionPropertyName(instanceName, "includeRoot")));
    if (!attr.IsValid()) return false;
    auto resolved = attr.Get();
    if (!resolved.found) return false;
    if (auto* value = resolved.value.Get<Bool>()) return *value;
    return false;
}

std::vector<Path> GetCollectionRelationshipTargets(
        const UsdPrim& prim,
        const Token& instanceName,
        std::string_view suffix) {
    auto rel = prim.GetRelationship(
        Token(CollectionPropertyName(instanceName, suffix)));
    if (!rel.IsValid()) return {};
    return rel.GetTargets();
}

struct CollectionAccumulator {
    std::vector<Path> paths;
    PathSet seen;

    void Add(const Path& path) {
        if (path.IsEmpty()) return;
        if (seen.insert(path).second) paths.push_back(path);
    }
};

void AddCollectionPrimSubtree(CollectionAccumulator& out,
                              const UsdPrim& prim,
                              bool includeProperties) {
    if (!prim.IsValid()) return;
    out.Add(prim.GetPath());
    if (includeProperties) {
        for (const auto& propName : prim.GetPropertyNames()) {
            Path propPath = prim.GetPath().AppendProperty(propName);
            if (!propPath.IsEmpty()) out.Add(propPath);
        }
    }
    for (const auto& child : prim.GetChildren()) {
        AddCollectionPrimSubtree(out, child, includeProperties);
    }
}

void AddExpandedCollectionTarget(CollectionAccumulator& out,
                                 const Stage* stage,
                                 const Path& target,
                                 CollectionExpansionRule rule) {
    if (target.IsEmpty()) return;

    if (rule == CollectionExpansionRule::ExplicitOnly || target.HasProperty()) {
        out.Add(target);
        return;
    }

    const bool includeProperties =
        rule == CollectionExpansionRule::ExpandPrimsAndProperties;
    if (!stage) {
        out.Add(target);
        return;
    }

    if (target.IsAbsoluteRoot()) {
        out.Add(target);
        for (const auto& rootPrim : stage->GetRootPrims()) {
            AddCollectionPrimSubtree(out, rootPrim, includeProperties);
        }
        return;
    }

    auto prim = stage->GetPrimAtPath(target);
    if (!prim.IsValid()) return;
    AddCollectionPrimSubtree(out, prim, includeProperties);
}

bool CollectionExcludesMember(const Path& excluded, const Path& member) {
    if (excluded.IsEmpty() || member.IsEmpty()) return false;
    if (excluded.HasProperty()) return member == excluded;

    const Path memberPrimPath = member.HasProperty()
        ? member.GetPrimPath()
        : member;
    return IsAncestorOrEqual(excluded, memberPrimPath);
}

bool CollectionExcludeTargetIsCollectionMarker(const Path& target) {
    return GetCollectionInstanceFromMarkerPath(target).has_value();
}

std::vector<Path> ApplyCollectionExcludes(
        const std::vector<Path>& included,
        const std::vector<Path>& excludes) {
    if (included.empty() || excludes.empty()) return included;

    std::vector<Path> result;
    result.reserve(included.size());
    for (const auto& member : included) {
        bool excluded = false;
        for (const auto& exclude : excludes) {
            if (CollectionExcludeTargetIsCollectionMarker(exclude)) continue;
            if (CollectionExcludesMember(exclude, member)) {
                excluded = true;
                break;
            }
        }
        if (!excluded) result.push_back(member);
    }
    return result;
}

std::string CollectionCycleKey(const UsdPrim& prim,
                               const Token& instanceName) {
    std::string key = prim.GetPath().GetText();
    key.push_back('.');
    key += "collection:";
    key += instanceName.GetString();
    return key;
}

void EvaluateCollectionInto(CollectionAccumulator& out,
                            const Stage* stage,
                            const UsdPrim& prim,
                            const Token& instanceName,
                            std::unordered_set<std::string>& activeCollections) {
    if (!prim.IsValid() || instanceName.IsEmpty()) return;

    const std::string cycleKey = CollectionCycleKey(prim, instanceName);
    if (!activeCollections.insert(cycleKey).second) return;

    const CollectionExpansionRule rule =
        GetCollectionExpansionRule(prim, instanceName);

    CollectionAccumulator local;
    if (GetCollectionIncludeRoot(prim, instanceName)) {
        AddExpandedCollectionTarget(
            local, stage, Path::AbsoluteRoot(), rule);
    }

    for (const auto& target :
         GetCollectionRelationshipTargets(prim, instanceName, "includes")) {
        if (auto targetInstance =
                GetCollectionInstanceFromMarkerPath(target)) {
            if (stage) {
                auto targetPrim = stage->GetPrimAtPath(target.GetPrimPath());
                EvaluateCollectionInto(local, stage, targetPrim,
                                       *targetInstance, activeCollections);
            }
            continue;
        }
        AddExpandedCollectionTarget(local, stage, target, rule);
    }

    const auto filtered = ApplyCollectionExcludes(
        local.paths,
        GetCollectionRelationshipTargets(prim, instanceName, "excludes"));
    for (const auto& path : filtered) out.Add(path);

    activeCollections.erase(cycleKey);
}

} // anonymous namespace

std::vector<Path> UsdPrim::ComputeCollectionMembership(
        const Token& instanceName) const {
    CollectionAccumulator out;
    std::unordered_set<std::string> activeCollections;
    EvaluateCollectionInto(out, stage_, *this, instanceName, activeCollections);
    return std::move(out.paths);
}

bool UsdPrim::IsCollectionMember(const Token& instanceName,
                                 const Path& path) const {
    if (path.IsEmpty()) return false;
    auto members = ComputeCollectionMembership(instanceName);
    return std::find(members.begin(), members.end(), path) != members.end();
}

bool Stage::IsInPrototypeNamespace(const Path& path) const {
    if (path.IsEmpty()) return false;
    for (const auto& protoRoot : prototypeRoots_) {
        if (IsAncestorOrEqual(protoRoot, path)) return true;
    }
    return false;
}

bool Stage::PopulationMaskIncludes(const Path& path) const {
    if (!hasPopulationMask_) return true;
    if (path.IsEmpty()) return false;
    if (path.IsAbsoluteRoot()) return true;
    if (IsInPrototypeNamespace(path)) return true;

    for (const auto& maskPath : populationMask_) {
        if (maskPath.IsAbsoluteRoot()) return true;
        if (IsAncestorOrEqual(path, maskPath)) {
            return true;
        }
    }
    return false;
}

void Stage::BuildPrototypeSubtree(const Path& protoPath, const Path& instancePath) {
    Path representative = graph_.GetInstanceRepresentative(instancePath);
    if (!representative.IsEmpty()) {
        BuildPrototypeSubtreeImpl(protoPath, representative, representative, protoPath);
        return;
    }
    BuildPrototypeSubtreeImpl(protoPath, instancePath, instancePath, protoPath);
}

namespace {
// Remap a path that was rooted at `origPrefix` (e.g. /InstanceHigh) so
// it is rooted at `newPrefix` (e.g. /__Prototype_0) instead, preserving
// any suffix. Returns `newPrefix` if the input equals the prefix
// exactly. Used to translate opinion-entry arc targets from instance
// namespace into prototype namespace.
Path RemapRoot(const Path& input, const Path& origPrefix, const Path& newPrefix) {
    const auto& inElems = GetCachedPathElements(input);
    const auto& prefElems = GetCachedPathElements(origPrefix);
    if (inElems.size() < prefElems.size()) return input;
    for (size_t i = 0; i < prefElems.size(); ++i) {
        if (inElems[i].name != prefElems[i].name) return input;
    }
    Path result = newPrefix;
    for (size_t i = prefElems.size(); i < inElems.size(); ++i) {
        result = result.AppendChild(inElems[i].name);
    }
    return result;
}
} // anonymous namespace

// Helper: build a filtered PrimIndex at `newTargetPath` from the entries
// of `srcIdx`, keeping only those that represent composition arcs rooted
// at or below `origInstancePath`. Strict-ancestor arcs (opinions that
// reach this prim via a parent's arc) are excluded — they're per-instance
// local overrides, not part of the shared prototype.
//
// For kept entries, the pathMapping's targetPrimPath is the arc's
// landing point in instance namespace (e.g. /InstanceHigh, or a
// descendant of it for internally-rooted arcs). That target is remapped
// from origInstancePath → origProtoPath to relocate the arc into
// prototype namespace — NOT simply set to newTargetPath, which would
// misalign MapToSource for any entry that came from an ancestor-level
// IndexDescendantPrims call (template entry propagates the arc's root
// target down into each descendant primIndex). Returns true iff the
// filtered index is non-empty and was inserted.
static bool BuildFilteredPrimIndex(CompositionGraph& graph,
                                    const PrimIndex* srcIdx,
                                    const Path& origInstancePath,
                                    const Path& origProtoPath,
                                    const Path& newTargetPath) {
    // Many consecutive entries from the same arc share the same source
    // pathMapping. Memoize the remapped pathMapping per unique source
    // mapping so we only allocate one fresh PathMapping per arc instead
    // of per entry.
    const PathMapping* lastSrc = nullptr;
    PathMappingPtr lastRemapped;

    PrimIndex filtered;
    for (const auto& entry : srcIdx->entries) {
        if (!IsAncestorOrEqual(origInstancePath, entry.pathMapping->targetPrimPath))
            continue;
        OpinionEntry remapped = entry;
        if (!remapped.pathMapping->isIdentity) {
            const PathMapping* src = remapped.pathMapping.get();
            if (src != lastSrc) {
                auto m = std::make_shared<PathMapping>(*src);
                m->targetPrimPath = RemapRoot(src->targetPrimPath,
                                              origInstancePath, origProtoPath);
                for (auto& extra : m->extraMappings) {
                    if (!extra.targetPrimPath.IsEmpty()) {
                        extra.targetPrimPath = RemapRoot(
                            extra.targetPrimPath,
                            origInstancePath,
                            origProtoPath);
                    }
                }
                lastRemapped = std::move(m);
                lastSrc = src;
            }
            remapped.pathMapping = lastRemapped;
        }
        if (!remapped.pathMapping->targetPrimPath.IsEmpty() &&
            IsAncestorOrEqual(newTargetPath, remapped.pathMapping->targetPrimPath)) {
            filtered.hasInstancingKeyEntries = true;
        }
        filtered.entries.push_back(std::move(remapped));
    }
    if (filtered.entries.empty()) return false;
    graph.SetPrimIndex(newTargetPath, std::move(filtered));
    graph.AddToChildIndex(newTargetPath);
    return true;
}

void Stage::BuildPrototypeSubtreeImpl(const Path& protoPath,
                                       const Path& instancePath,
                                       const Path& origInstancePath,
                                       const Path& origProtoPath) {
    const PrimIndex* instanceIdx = graph_.GetPrimIndex(instancePath);
    if (!instanceIdx) return;

    if (!BuildFilteredPrimIndex(graph_, instanceIdx,
                                 origInstancePath, origProtoPath, protoPath))
        return;

    // Recurse into composed children, filtered against the original instance.
    const auto& childNames = graph_.GetComposedChildNames(instancePath);
    for (const auto& childName : childNames) {
        Path instanceChildPath = instancePath.AppendChild(childName);
        Path protoChildPath = protoPath.AppendChild(childName);

        const PrimIndex* childIdx = graph_.GetPrimIndex(instanceChildPath);
        if (!childIdx || childIdx->entries.empty()) continue;

        if (!BuildFilteredPrimIndex(graph_, childIdx,
                                     origInstancePath, origProtoPath, protoChildPath))
            continue;

        BuildPrototypeSubtreeImpl(protoChildPath, instanceChildPath,
                                   origInstancePath, origProtoPath);
    }
}

void Stage::Populate() {
    populatedPrims_.clear();
    prototypePaths_.clear();
    instanceToPrototype_.clear();
    prototypeToInstances_.clear();
    prototypeToInstancesDirty_ = true;
    prototypeRoots_.clear();
    const size_t composedChildEntryCount = graph_.GetComposedChildEntryCount();
    populatedPrims_.reserve(composedChildEntryCount);

    // Map from instancing key to prototype path
    std::unordered_map<InstancingKey, Path, InstancingKey::Hash> keyToPrototype;
    int prototypeCounter = 0;

    bool timing = AttrTimingEnabled();
    long long ns_children = 0, ns_opinion = 0, ns_push = 0, ns_instancing = 0;
    long long ns_make_key = 0, ns_key_lookup = 0, ns_build_proto = 0, ns_reuse = 0;
    long long ns_rep_lookup = 0;
    long long calls = 0, instanceableCalls = 0, instancingKeyBuilt = 0;
    long long protosCreated = 0, protosReused = 0;
    long long representativeReused = 0;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto addNs = [](long long& bucket, auto a, auto b) {
        bucket += std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    };

    bool hasLayerStackRelocates = false;
    for (const auto& stack : graph_.layerStacks) {
        if (!stack.relocates.empty()) {
            hasLayerStackRelocates = true;
            break;
        }
    }
    const bool singleLayerLocalOnly =
        graph_.arcOriginLayerCount == 1 &&
        graph_.GetNumLayers() == 1 &&
        graph_.primIndices.empty() &&
        !hasLayerStackRelocates;

    // Build a depth-first traversal of populated prims
    auto traverse = [&](auto&& self, const Path& parentPath) -> void {
        // O(1) lookup of direct child prim names from the composed index
        auto tChild0 = timing ? now() : std::chrono::steady_clock::time_point{};
        auto indexedChildren = graph_.GetComposedChildren(parentPath);
        const auto& indexedNames = *indexedChildren.names;
        const auto& indexedPaths = *indexedChildren.paths;
        const auto& indexedPrimIndices = *indexedChildren.primIndices;
        const auto& indexedChildHasChildren = *indexedChildren.childHasChildren;
        const PrimIndex* parentIdx = parentPath.IsAbsoluteRoot() ?
            nullptr : graph_.GetPrimIndex(parentPath);
        const Spec* parentSpec = parentPath.IsAbsoluteRoot() ?
            &graph_.composedLayerSpec : nullptr;
        std::vector<Token> orderedChildNames;
        const std::vector<Token>* childNames = &indexedNames;
        bool hasPrimOrder = false;
        if (!indexedNames.empty()) {
            orderedChildNames.assign(indexedNames.begin(), indexedNames.end());
            hasPrimOrder = SortAndApplyComposedPrimOrder(
                orderedChildNames, &graph_, parentIdx, parentPath, parentSpec);
        }
        if (hasPrimOrder) {
            childNames = &orderedChildNames;
        }
        const bool useIndexedPaths =
            !hasPrimOrder && indexedPaths.size() == indexedNames.size();
        const bool useIndexedPrimIndices =
            !hasPrimOrder && indexedPrimIndices.size() == indexedNames.size();
        const bool useIndexedChildHasChildren =
            !hasPrimOrder && indexedChildHasChildren.size() == indexedNames.size();
        const size_t childCount =
            useIndexedPaths ? indexedPaths.size() : childNames->size();
        if (timing) addNs(ns_children, tChild0, now());

        for (size_t childI = 0; childI < childCount; ++childI) {
            if (timing) calls++;
            Path childPath = useIndexedPaths
                ? indexedPaths[childI]
                : parentPath.AppendChild((*childNames)[childI]);
            if (!PopulationMaskIncludes(childPath)) continue;

            if (singleLayerLocalOnly) {
                auto tOp0 = timing ? now() : std::chrono::steady_clock::time_point{};
                const Spec* localSpec = graph_.GetLayer(0).GetPrimSpec(childPath);
                if (timing) addNs(ns_opinion, tOp0, now());
                if (!localSpec) continue;
                if (localSpec->GetSpecifier() == Specifier::Over) continue;
                if (!localSpec->GetActive()) continue;

                auto tPush0 = timing ? now() : std::chrono::steady_clock::time_point{};
                populatedPrims_.push_back({childPath, nullptr, 0, Path()});
                if (timing) addNs(ns_push, tPush0, now());
                if (!useIndexedChildHasChildren || indexedChildHasChildren[childI])
                    self(self, childPath);
                continue;
            }

            const PrimIndex* childIdx = useIndexedPrimIndices
                ? indexedPrimIndices[childI]
                : nullptr;
            if (!childIdx) childIdx = graph_.GetPrimIndex(childPath);
            if (!childIdx || childIdx->entries.empty()) continue;

            auto tOp0 = timing ? now() : std::chrono::steady_clock::time_point{};
            // Fuse three per-prim walks (composed specifier, active,
            // instanceable) into one opinion-entry walk. All three use
            // "first authored opinion wins" semantics, so we early-exit
            // once each has been resolved.
            bool haveSpec = false, haveActive = false, haveInst = false;
            bool haveStrongestSpec = false;
            size_t strongestLayerIndex = 0;
            Path strongestSourcePath;
            const Value* activeVal = nullptr;
            const Value* instVal = nullptr;
            for (const auto& entry : childIdx->entries) {
                if (haveSpec && haveActive && haveInst) break;
                const Layer& layer = graph_.GetLayer(entry.layerIndex);
                Path sourcePath = entry.pathMapping->isIdentity ? childPath :
                    OpinionSourcePrimPath(entry, childPath);
                if (sourcePath.IsEmpty()) continue;
                const auto* s = layer.GetPrimSpec(sourcePath);
                if (!s) continue;
                if (!haveStrongestSpec) {
                    strongestLayerIndex = entry.layerIndex;
                    if (!entry.pathMapping->isIdentity)
                        strongestSourcePath = sourcePath;
                    haveStrongestSpec = true;
                }
                if (!haveSpec && s->GetSpecifier() != Specifier::Over)
                    haveSpec = true;
                if (!haveActive) {
                    if (auto* v = s->GetField(FieldNames::active)) {
                        activeVal = v; haveActive = true;
                    }
                }
                if (!haveInst) {
                    if (auto* v = s->GetField(FieldNames::instanceable)) {
                        instVal = v; haveInst = true;
                    }
                }
            }
            if (timing) addNs(ns_opinion, tOp0, now());

            // Skip Over-only prims — dangling overrides aren't populated.
            if (!haveSpec) continue;
            if (!haveStrongestSpec) continue;

            // Active: default true; authored false/0 skips the prim.
            if (!IsActiveValueEnabled(activeVal)) continue;

            // Instanceable (spec Section 11.3): default false.
            bool instanceable = false;
            if (instVal) {
                if (auto* b = instVal->Get<bool>()) instanceable = *b;
            }

            auto tInst0 = timing ? now() : std::chrono::steady_clock::time_point{};
            if (instanceable && graph_.arcOriginLayerCount > 0 &&
                childIdx->hasInstancingKeyEntries) {
                if (timing) instanceableCalls++;
                auto tRep0 = timing ? now() : std::chrono::steady_clock::time_point{};
                Path representative = graph_.GetInstanceRepresentative(childPath);
                if (!representative.IsEmpty()) {
                    auto protoIt = instanceToPrototype_.find(representative);
                    if (protoIt != instanceToPrototype_.end()) {
                        // Copy out before the [] inserts below — those may grow
                        // instanceToPrototype_ and invalidate `protoIt` under
                        // any flat hash map (incl. ankerl::unordered_dense's
                        // non-segmented variants if we ever switch). Cheap
                        // pointer-handle copy; safer pattern with flat maps.
                        Path protoPath = protoIt->second;
                        if (timing) {
                            protosReused++;
                            representativeReused++;
                            addNs(ns_rep_lookup, tRep0, now());
                        }
                        populatedPrims_.push_back(
                            {childPath, childIdx, strongestLayerIndex,
                             strongestSourcePath});
                        if (timing) addNs(ns_instancing, tInst0, now());
                        continue;
                    }
                }
                if (timing) addNs(ns_rep_lookup, tRep0, now());

                auto tKey0 = timing ? now() : std::chrono::steady_clock::time_point{};
                auto key = MakeInstancingKey(childIdx, childPath, graph_);
                if (timing) addNs(ns_make_key, tKey0, now());
                if (!key.empty) {
                    if (timing) instancingKeyBuilt++;
                    auto tLook0 = timing ? now() : std::chrono::steady_clock::time_point{};
                    auto it = keyToPrototype.find(key);
                    if (timing) addNs(ns_key_lookup, tLook0, now());
                    if (it == keyToPrototype.end()) {
                        if (timing) protosCreated++;
                        // First instance with this key — create prototype
                        Token protoName("__Prototype_" + std::to_string(prototypeCounter++));
                        Path protoPath = Path::AbsoluteRoot().AppendChild(protoName);
                        prototypePaths_.push_back(protoPath);
                        prototypeRoots_.insert(protoPath);
                        keyToPrototype[key] = protoPath;
                        auto tBuild0 = timing ? now() : std::chrono::steady_clock::time_point{};
                        BuildPrototypeSubtree(protoPath, childPath);
                        if (timing) addNs(ns_build_proto, tBuild0, now());
                        instanceToPrototype_[childPath] = protoPath;
                    } else {
                        if (timing) protosReused++;
                        auto tReuse0 = timing ? now() : std::chrono::steady_clock::time_point{};
                        // Reuse existing prototype
                        instanceToPrototype_[childPath] = it->second;
                        if (timing) addNs(ns_reuse, tReuse0, now());
                    }

                    // Add instance to populated prims but do NOT recurse into children
                    populatedPrims_.push_back(
                        {childPath, childIdx, strongestLayerIndex,
                         strongestSourcePath});
                    if (timing) addNs(ns_instancing, tInst0, now());
                    continue;
                }
            }
            if (timing) addNs(ns_instancing, tInst0, now());

            auto tPush0 = timing ? now() : std::chrono::steady_clock::time_point{};
            populatedPrims_.push_back(
                {childPath, childIdx, strongestLayerIndex, strongestSourcePath});
            if (timing) addNs(ns_push, tPush0, now());
            if (!useIndexedChildHasChildren || indexedChildHasChildren[childI])
                self(self, childPath);
        }
    };

    traverse(traverse, Path::AbsoluteRoot());
    populateDirty_ = false;

    if (timing) {
        auto ms = [](long long ns) { return ns / 1e6; };
        std::fprintf(stderr,
            "      [populate] prims=%lld  instanceable=%lld  "
            "protos created=%lld reused=%lld  representative reused=%lld\n",
            calls, instanceableCalls, protosCreated, protosReused,
            representativeReused);
        std::fprintf(stderr,
            "      [populate]   children=%.1f ms  opinion=%.1f ms  push=%.1f ms\n",
            ms(ns_children), ms(ns_opinion), ms(ns_push));
        std::fprintf(stderr,
            "      [populate]   make_key=%.1f ms  key_lookup=%.1f ms  "
            "rep_lookup=%.1f ms  build_proto=%.1f ms  reuse=%.1f ms  "
            "(instancing total=%.1f)\n",
            ms(ns_make_key), ms(ns_key_lookup), ms(ns_rep_lookup),
            ms(ns_build_proto), ms(ns_reuse), ms(ns_instancing));
    }
}

// --- Stage instancing queries ---

bool Stage::IsInstance(const Path& path) const {
    EnsurePopulated();
    return !GetPrototypePath(path).IsEmpty();
}

bool Stage::IsPrototypeRoot(const Path& path) const {
    EnsurePopulated();
    return prototypeRoots_.count(path) > 0;
}

Path Stage::GetPrototypePath(const Path& instancePath) const {
    EnsurePopulated();
    auto it = instanceToPrototype_.find(instancePath);
    if (it != instanceToPrototype_.end()) return it->second;
    Path representative = graph_.GetInstanceRepresentative(instancePath);
    if (representative.IsEmpty()) return {};
    auto repIt = instanceToPrototype_.find(representative);
    return repIt != instanceToPrototype_.end() ? repIt->second : Path();
}

const std::vector<Path>& Stage::GetPrototypePaths() const {
    EnsurePopulated();
    return prototypePaths_;
}

const std::vector<Path>& Stage::GetInstancePaths(const Path& prototypePath) const {
    EnsurePopulated();
    EnsurePrototypeInstanceIndex();
    auto it = prototypeToInstances_.find(prototypePath);
    return it != prototypeToInstances_.end() ? it->second : s_emptyPaths;
}

void Stage::EnsurePrototypeInstanceIndex() const {
    if (!prototypeToInstancesDirty_) return;
    prototypeToInstances_.clear();
    if (!instanceToPrototype_.empty()) {
        prototypeToInstances_.reserve(prototypeRoots_.size());
        for (const auto& record : populatedPrims_) {
            Path protoPath = GetPrototypePath(record.path);
            if (protoPath.IsEmpty()) continue;
            prototypeToInstances_[protoPath].push_back(record.path);
        }
    }
    prototypeToInstancesDirty_ = false;
}

UsdPrim Stage::GetPseudoRoot() const {
    auto* self = const_cast<Stage*>(this);
    return UsdPrim(Path::AbsoluteRoot(), "", &graph_.composedLayerSpec,
                   &self->graph_, nullptr, size_t(0), this);
}

UsdPrim Stage::GetDefaultPrim() const {
    auto dp = graph_.composedLayerSpec.GetDefaultPrim();
    if (dp.GetString().empty()) return {};
    return GetPrimAtPath(Path::AbsoluteRoot().AppendChild(dp));
}

std::vector<UsdPrim> Stage::GetRootPrims() const {
    return GetPseudoRoot().GetChildren();
}

UsdPrim Stage::GetPrimAtPath(const Path& path) const {
    EnsurePopulated();
    if (!PopulationMaskIncludes(path)) return {};
    auto* self = const_cast<Stage*>(this);
    const PrimIndex* idx = graph_.GetPrimIndex(path);
    if (!idx || idx->entries.empty()) return {};

    const Spec* strongestSpec = GetStrongestPrimSpec(&graph_, idx, path);
    if (!strongestSpec) return {};

    if (!PrimAndAncestorsAreDefined(&graph_, idx, path, strongestSpec))
        return {};
    if (!PrimAndAncestorsAreActive(&graph_, idx, path, strongestSpec))
        return {};

    return UsdPrim(path, path.GetName(), strongestSpec,
                   &self->graph_, idx, size_t(0), this);
}

bool Stage::HasPrimAtPath(const Path& path) const {
    return GetPrimAtPath(path).IsValid();
}

std::vector<UsdPrim> Stage::Traverse() const {
    EnsurePopulated();
    if (AttrTimingEnabled()) g_attrStats.reset();
    auto* self = const_cast<Stage*>(this);
    std::vector<UsdPrim> result;
    result.reserve(populatedPrims_.size());
    for (const auto& record : populatedPrims_) {
        const Path& path = record.path;
        const PrimIndex* idx = record.primIndex;
        const Spec* strongestSpec = nullptr;
        if (record.strongestLayerIndex < graph_.GetNumLayers()) {
            const Path& sourcePath = record.strongestSourcePath.IsEmpty()
                ? path
                : record.strongestSourcePath;
            strongestSpec = graph_.GetLayer(record.strongestLayerIndex)
                                 .GetPrimSpec(sourcePath);
        }
        if (!idx && !strongestSpec)
            idx = graph_.GetPrimIndex(path);
        if (idx && idx->entries.empty()) continue;
        // A layer edit can leave population ordering valid while changing
        // spec storage. Fall back to the opinion walk rather than returning
        // a handle with a stale or missing spec.
        if (!strongestSpec && idx) {
            strongestSpec = GetStrongestPrimSpec(&graph_, idx, path);
        }
        if (strongestSpec) {
            result.emplace_back(path, path.GetName(), strongestSpec,
                               &self->graph_, idx, record.strongestLayerIndex,
                               this);
        }
    }
    return result;
}

double Stage::GetTimeCodesPerSecond() const {
    return graph_.composedLayerSpec.GetTimeCodesPerSecond();
}

double Stage::GetFramesPerSecond() const {
    return graph_.composedLayerSpec.GetFramesPerSecond();
}

double Stage::GetStartTimeCode() const {
    return graph_.composedLayerSpec.GetStartTimeCode();
}

double Stage::GetEndTimeCode() const {
    return graph_.composedLayerSpec.GetEndTimeCode();
}

// ============================================================
// FlattenStage
// ============================================================

// Fields that are composition operators — stripped during flatten.
static bool IsCompositionField(const std::string& name) {
    return name == "references" || name == "payload"
        || name == "inheritPaths" || name == "specializes"
        || name == "subLayers" || name == "subLayerOffsets"
        || name == "layerRelocates";
}

// Fields that are structural ordering hints — rebuilt by flatten.
static bool IsStructuralField(const std::string& name) {
    return name == "primChildren" || name == "propertyChildren"
        || name == "variantSetChildren" || name == "variantChildren";
}

static bool IsSampledLayerValueSourcePrimField(const std::string& name) {
    return name == FieldNames::clips.GetString() ||
           name == FieldNames::clipSets.GetString();
}

static double ForwardRetimedTime(double time, const Retiming& retiming) {
    return time * retiming.scale + retiming.offset;
}

static std::string FlattenTimeKey(double time) {
    std::ostringstream oss;
    oss << time;
    return oss.str();
}

static std::vector<double> SortUniqueSampleTimes(std::vector<double> times) {
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end()), times.end());
    return times;
}

static void CollectSplineKnotTimes(const Spec& spec,
                                   const Retiming& retiming,
                                   std::vector<double>& times) {
    const auto* field = spec.GetField(FieldNames::spline);
    if (!field || field->IsBlock()) return;
    const auto* spline = field->Get<Spline>();
    if (!spline) return;
    for (const auto& knot : spline->knots) {
        times.push_back(ForwardRetimedTime(knot.time, retiming));
    }
}

static void CollectTimeSampleTimes(const Spec& spec,
                                   const Retiming& retiming,
                                   std::vector<double>& times) {
    const auto* field = spec.GetField(FieldNames::timeSamples);
    if (!field || field->IsBlock()) return;
    const auto* samples = field->Get<Dictionary>();
    if (!samples) return;
    for (const auto& [key, _value] : *samples) {
        times.push_back(ForwardRetimedTime(std::stod(key), retiming));
    }
}

static void CollectClipTimeArray(const Dictionary& clipSet,
                                 const char* fieldName,
                                 const Retiming& retiming,
                                 std::vector<double>& times) {
    auto it = clipSet.find(fieldName);
    if (it == clipSet.end()) return;
    const auto* values = it->second.Get<std::vector<GfVec2d>>();
    if (!values) return;
    for (const auto& pair : *values) {
        times.push_back(ForwardRetimedTime(pair[0], retiming));
    }
}

static void CollectClipSampleTimes(const CompositionGraph& graph,
                                   const Spec& primSpec,
                                   const OpinionEntry& entry,
                                   std::vector<double>& times) {
    const auto* clipsField = primSpec.GetField(FieldNames::clips);
    if (!clipsField || clipsField->IsBlock()) return;

    if (const auto* clips = clipsField->Get<Dictionary>()) {
        for (const auto& [_name, value] : *clips) {
            const auto* clipSet = value.Get<Dictionary>();
            if (!clipSet) continue;
            CollectClipTimeArray(*clipSet, "active", entry.retiming, times);
            CollectClipTimeArray(*clipSet, "times", entry.retiming, times);
        }
    }

    auto clipSets = MaterializeClipSetsForLayer(
        primSpec, graph.layerPaths[entry.layerIndex], entry.layerIndex,
        graph.resolver);
    for (const auto& clipSet : clipSets) {
        for (const auto& [stageTime, _assetIndex] : clipSet.active) {
            times.push_back(ForwardRetimedTime(stageTime, entry.retiming));
        }
        for (const auto& [stageTime, _clipTime] : clipSet.times) {
            times.push_back(ForwardRetimedTime(stageTime, entry.retiming));
        }
    }
}

static std::vector<double> CollectFlattenSampleTimes(
        const CompositionGraph& graph,
        const PrimIndex* primIdx,
        const Path& attrPath) {
    std::vector<double> times;
    if (!primIdx) return times;

    for (const auto& entry : primIdx->entries) {
        const Layer& layer = graph.GetLayer(entry.layerIndex);
        Path sourcePath = OpinionSourcePath(entry, attrPath);
        if (sourcePath.IsEmpty()) continue;

        if (const auto* spec = layer.GetAttributeSpec(sourcePath)) {
            CollectTimeSampleTimes(*spec, entry.retiming, times);
            CollectSplineKnotTimes(*spec, entry.retiming, times);
        }

        if (const auto* primSpec = layer.GetPrimSpec(sourcePath.GetPrimPath())) {
            CollectClipSampleTimes(graph, *primSpec, entry, times);
        }
    }

    return SortUniqueSampleTimes(std::move(times));
}

static std::optional<Value> ResolveFlattenDefaultValue(
        const CompositionGraph& graph,
        const PrimIndex* primIdx,
        const Path& attrPath,
        const UsdAttribute& attr) {
    if (primIdx) {
        for (const auto& entry : primIdx->entries) {
            const Layer& layer = graph.GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePath(entry, attrPath);
            if (sourcePath.IsEmpty()) continue;
            const auto* spec = layer.GetAttributeSpec(sourcePath);
            if (!spec) continue;

            const auto* authored = spec->GetField(FieldNames::defaultValue);
            if (!authored) continue;
            if (authored->IsBlock()) {
                if (const auto* fallback = attr.GetFallback())
                    return *fallback;
                return std::nullopt;
            }

            Value copy = *authored;
            if (graph.resolver) {
                ResolveAssetValueOrDict(copy, graph.layerPaths[entry.layerIndex],
                                        graph.resolver);
            }
            return copy;
        }
    }

    if (const auto* fallback = attr.GetFallback())
        return *fallback;
    return std::nullopt;
}

Layer FlattenStage(const Stage& stage) {
    Layer flat;

    if (!stage.IsValid()) return flat;

    const auto& graph = stage.GetGraph();

    // Copy resolved root layer metadata, stripping composition arcs.
    // For in-memory stages the composedLayerSpec may be empty, so also
    // check the root layer's layer spec.
    {
        Spec layerSpec(SpecType::Layer);
        auto copyFields = [&](const Spec& src) {
            for (const auto& [name, val] : src.GetFields()) {
                if (IsCompositionField(name) || IsStructuralField(name))
                    continue;
                if (!layerSpec.HasField(Token(name))) {
                    layerSpec.SetField(Token(name), val);
                }
            }
        };
        copyFields(graph.composedLayerSpec);
        if (!graph.layers.empty()) {
            copyFields(graph.GetRootLayer().GetLayerSpec());
        }
        flat.GetLayerSpec() = std::move(layerSpec);
    }

    // Walk all populated prims
    auto prims = stage.Traverse();
    for (const auto& prim : prims) {
        Path primPath = prim.GetPath();
        const PrimIndex* primIdx = graph.GetPrimIndex(primPath);
        if (!primIdx) continue;

        // Build the flattened prim spec by walking the opinion stack
        Spec flatPrimSpec(SpecType::Prim);

        // Specifier: first non-Over wins
        flatPrimSpec.SetSpecifier(prim.GetSpecifier());

        // Collect the union of prim metadata field names across opinions
        // (preserving first-seen order for stable output). Per-field
        // resolution depends on type: Dictionary fields combine across
        // the opinion stack (spec §12.2.5 + §6.6.2.1); non-dict fields
        // use first-authored-wins semantics.
        std::unordered_set<std::string> seenFields;
        std::vector<std::string> fieldOrder;
        for (const auto& entry : primIdx->entries) {
            const Layer& layer = graph.GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePrimPath(entry, primPath);
            if (sourcePath.IsEmpty()) continue;

            const auto* spec = layer.GetPrimSpec(sourcePath);
            if (!spec) continue;

            for (const auto& [name, _val] : spec->GetFields()) {
                if (IsCompositionField(name) || IsStructuralField(name) ||
                    IsSampledLayerValueSourcePrimField(name))
                    continue;
                if (name == "specifier") continue;
                if (seenFields.insert(name).second)
                    fieldOrder.push_back(name);
            }
        }

        for (const auto& name : fieldOrder) {
            const auto* desc = FieldRegistry::Instance().Find(Token(name));
            if (desc && desc->type == TypeId::Dictionary) {
                auto combined = ResolveDictionaryField(
                    &graph, primIdx, primPath, Token(name), graph.resolver);
                if (combined)
                    flatPrimSpec.SetField(Token(name), std::move(*combined));
                continue;
            }
            // ListOp metadata combines across opinions per spec §12.2.6.
            // Dispatch on the actual stored element type:
            //   ListOp<Path>         — connectionPaths, targetPaths
            //   ListOp<Token>        — apiSchemas (§13.2.1.2)
            //   ListOp<std::string>  — clipSets (§13.2.1.2.2), variantSetNames
            if (desc && desc->isListOp) {
                if (auto p = ResolveListOpField<Path>(graph, primIdx, primPath,
                                                      Token(name), SpecType::Prim)) {
                    flatPrimSpec.SetField(Token(name), Value(std::move(*p)));
                    continue;
                }
                if (auto t = ResolveListOpField<Token>(graph, primIdx,
                                                        primPath, Token(name),
                                                        SpecType::Prim)) {
                    flatPrimSpec.SetField(Token(name), Value(std::move(*t)));
                    continue;
                }
                if (auto s = ResolveListOpField<std::string>(graph, primIdx,
                                                              primPath, Token(name),
                                                              SpecType::Prim)) {
                    flatPrimSpec.SetField(Token(name), Value(std::move(*s)));
                    continue;
                }
                continue;
            }
            // Non-dict, non-listop field: first-authored wins.
            for (const auto& entry : primIdx->entries) {
                const Layer& layer = graph.GetLayer(entry.layerIndex);
                Path sourcePath = OpinionSourcePrimPath(entry, primPath);
                if (sourcePath.IsEmpty()) continue;
                const auto* spec = layer.GetPrimSpec(sourcePath);
                if (!spec) continue;
                const auto* val = spec->GetField(Token(name));
                if (!val) continue;
                // Resolve asset path values in metadata (spec §5).
                if (graph.resolver && val->GetTypeId() == TypeId::Asset) {
                    Value copy = *val;
                    ResolveAssetValueOrDict(copy, graph.layerPaths[entry.layerIndex],
                                            graph.resolver);
                    flatPrimSpec.SetField(Token(name), std::move(copy));
                } else {
                    flatPrimSpec.SetField(Token(name), *val);
                }
                break;
            }
        }
        flat.SetSpec(primPath, std::move(flatPrimSpec));

        // Flatten each property (only those authored in at least one layer)
        UsdPrim valuePrim = stage.GetPrimAtPath(primPath);
        auto propNames = prim.GetPropertyNames();
        for (const auto& propName : propNames) {
            Path propPath = primPath.AppendProperty(propName);

            // Find the first authored spec across layers to determine type
            // and skip schema-only properties with no authored data
            bool isRelationship = false;
            bool hasAuthoredSpec = false;
            for (const auto& entry : primIdx->entries) {
                const Layer& layer = graph.GetLayer(entry.layerIndex);
                Path sourcePath = OpinionSourcePath(entry, propPath);
                if (sourcePath.IsEmpty()) continue;
                const auto* spec = layer.GetSpec(sourcePath);
                if (!spec) continue;
                hasAuthoredSpec = true;
                isRelationship = (spec->GetType() == SpecType::Relationship);
                break;
            }
            if (!hasAuthoredSpec) continue;

            if (isRelationship) {
                // Flatten relationship: walk opinion stack for fields
                Spec flatRelSpec(SpecType::Relationship);
                for (const auto& entry : primIdx->entries) {
                    const Layer& layer = graph.GetLayer(entry.layerIndex);
                    Path sourcePath = OpinionSourcePath(entry, propPath);
                    if (sourcePath.IsEmpty()) continue;
                    const auto* spec = layer.GetRelationshipSpec(sourcePath);
                    if (!spec) continue;
                    for (const auto& [name, val] : spec->GetFields()) {
                        if (name == FieldNames::targetPaths)
                            continue;
                        if (IsCompositionField(name) || IsStructuralField(name))
                            continue;
                        if (!flatRelSpec.HasField(Token(name))) {
                            // Resolve asset path values in relationship metadata (spec §5)
                            if (graph.resolver &&
                                (val.GetTypeId() == TypeId::Asset ||
                                 val.GetTypeId() == TypeId::Dictionary)) {
                                Value copy = val;
                                ResolveAssetValueOrDict(copy, graph.layerPaths[entry.layerIndex],
                                                        graph.resolver);
                                flatRelSpec.SetField(Token(name), std::move(copy));
                            } else {
                                flatRelSpec.SetField(Token(name), val);
                            }
                        }
                    }
                }
                auto targets = prim.GetRelationship(propName).GetTargets();
                if (!targets.empty()) {
                    flatRelSpec.SetField(
                        FieldNames::targetPaths,
                        Value(ListOp<Path>::CreateExplicit(std::move(targets))));
                }
                flat.SetSpec(propPath, std::move(flatRelSpec));
            } else {
                // Flatten attribute: emit the sampled value artifact. The
                // dynamic value fields are resolved below rather than copied.
                Spec flatAttrSpec(SpecType::Attribute);

                // Walk opinion stack for scalar fields (typeName, variability,
                // custom, metadata). First authored wins.
                for (const auto& entry : primIdx->entries) {
                    const Layer& layer = graph.GetLayer(entry.layerIndex);
                    Path sourcePath = OpinionSourcePath(entry, propPath);
                    if (sourcePath.IsEmpty()) continue;
                    const auto* spec = layer.GetAttributeSpec(sourcePath);
                    if (!spec) continue;

                    for (const auto& [name, val] : spec->GetFields()) {
                        if (name == FieldNames::defaultValue.GetString() ||
                            name == FieldNames::timeSamples.GetString() ||
                            name == FieldNames::spline.GetString()) {
                            continue;
                        }
                        if (name == FieldNames::connectionPaths.GetString())
                            continue;
                        if (IsCompositionField(name) || IsStructuralField(name))
                            continue;
                        if (!flatAttrSpec.HasField(Token(name))) {
                            // Resolve asset path values in attribute fields (spec §2/§5)
                            if (graph.resolver &&
                                (val.GetTypeId() == TypeId::Asset ||
                                 val.GetTypeId() == TypeId::Dictionary)) {
                                Value copy = val;
                                ResolveAssetValueOrDict(copy, graph.layerPaths[entry.layerIndex],
                                                        graph.resolver);
                                flatAttrSpec.SetField(Token(name), std::move(copy));
                            } else {
                                flatAttrSpec.SetField(Token(name), val);
                            }
                        }
                    }
                }

                auto attr = valuePrim.IsValid()
                    ? valuePrim.GetAttribute(propName)
                    : prim.GetAttribute(propName);
                if (auto def = ResolveFlattenDefaultValue(graph, primIdx, propPath, attr)) {
                    flatAttrSpec.SetField(FieldNames::defaultValue, std::move(*def));
                }

                auto sampleTimes = CollectFlattenSampleTimes(graph, primIdx, propPath);
                if (!sampleTimes.empty()) {
                    Dictionary sampled;
                    for (double t : sampleTimes) {
                        auto resolved = attr.Get(UsdTimeCode(t));
                        if (!resolved.found || resolved.value.IsBlock()) continue;
                        sampled[FlattenTimeKey(t)] = std::move(resolved.value);
                    }
                    if (!sampled.empty()) {
                        flatAttrSpec.SetField(FieldNames::timeSamples,
                                              Value(std::move(sampled)));
                    }
                }

                auto connections = attr.GetConnections();
                if (!connections.empty()) {
                    flatAttrSpec.SetField(
                        FieldNames::connectionPaths,
                        Value(ListOp<Path>::CreateExplicit(
                            std::move(connections))));
                }

                flat.SetSpec(propPath, std::move(flatAttrSpec));
            }
        }
    }

    return flat;
}

} // namespace nanousd
