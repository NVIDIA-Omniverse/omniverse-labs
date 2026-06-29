// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "path.h"
#include "value.h"

#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nanousd {

class DecodeBacking;  // include "decode_backing.h" to use

// --- Spec forms per spec Section 7.3.1 ---

enum class SpecType {
    Layer,
    Prim,
    Attribute,
    Relationship,
    VariantSet,
    Variant,
};

// ============================================================
// Field name constants
// ============================================================
// Per spec Section 7.4, metadata fields are identified by name. The same
// field name must have the same type everywhere it appears.  These Token
// constants give every core field a canonical identity that can be compared
// in O(1).

namespace FieldNames {

// --- Hierarchy fields ---
inline const Token primChildren      {"primChildren"};
inline const Token propertyChildren  {"propertyChildren"};
inline const Token variantSetChildren{"variantSetChildren"};
inline const Token variantChildren   {"variantChildren"};

// --- Composition fields ---
inline const Token subLayers         {"subLayers"};
inline const Token subLayerOffsets   {"subLayerOffsets"};
inline const Token defaultPrim       {"defaultPrim"};
inline const Token layerRelocates    {"layerRelocates"};
inline const Token references        {"references"};
inline const Token payload           {"payload"};
inline const Token inheritPaths      {"inheritPaths"};
inline const Token specializes       {"specializes"};
inline const Token variantSetNames   {"variantSetNames"};
inline const Token variantSelection  {"variantSelection"};

// --- Population fields ---
inline const Token specifier         {"specifier"};
inline const Token typeName          {"typeName"};
inline const Token active            {"active"};
inline const Token instanceable      {"instanceable"};
inline const Token kind              {"kind"};
inline const Token primOrder         {"primOrder"};
inline const Token propertyOrder     {"propertyOrder"};
inline const Token apiSchemas        {"apiSchemas"};
inline const Token fallbackPrimTypes {"fallbackPrimTypes"};

// --- Value fields ---
inline const Token custom            {"custom"};
inline const Token variability       {"variability"};
inline const Token defaultValue      {"default"};
inline const Token timeSamples       {"timeSamples"};
inline const Token connectionPaths   {"connectionPaths"};
inline const Token spline            {"spline"};
inline const Token allowedTokens     {"allowedTokens"};
inline const Token colorSpace        {"colorSpace"};
inline const Token targetPaths       {"targetPaths"};

// --- Timing fields ---
inline const Token timeCodesPerSecond{"timeCodesPerSecond"};
inline const Token framesPerSecond   {"framesPerSecond"};
inline const Token startTimeCode     {"startTimeCode"};
inline const Token endTimeCode       {"endTimeCode"};

// --- User interface fields ---
inline const Token displayName       {"displayName"};
inline const Token displayGroup      {"displayGroup"};
inline const Token displayGroupOrder {"displayGroupOrder"};
inline const Token hidden            {"hidden"};
inline const Token documentation     {"documentation"};

// --- User fields ---
inline const Token customData        {"customData"};
inline const Token customLayerData   {"customLayerData"};
inline const Token assetInfo         {"assetInfo"};
inline const Token comment           {"comment"};

// --- Value clips (spec §12.3.4) ---
// clips is a prim-level Dictionary keyed by clip-set name; each value
// is itself a Dictionary of per-clip-set metadata (assetPaths, active,
// times, primPath, manifestAssetPath, template*, etc.). Composes
// across the opinion stack per §12.2.5 via dict combining, then
// consumed at attribute value resolution time per §12.3.4.5.
// clipSets is a ListOp<string> that names the search order of clip
// sets; combines across opinions per §12.2.6.
inline const Token clips             {"clips"};
inline const Token clipSets          {"clipSets"};

} // namespace FieldNames

// ============================================================
// FieldDescriptor — describes one core metadata field
// ============================================================

struct FieldDescriptor {
    Token name;
    TypeId type       = TypeId::Unknown;
    bool isRequired   = false;
    // Which spec forms support this field (bitmask-like set)
    std::unordered_set<SpecType> specForms;
    // True when the field's value is a list operation that participates
    // in cross-opinion combining per spec §12.2.6. Composition-input
    // listops (references / payload / inheritPaths / specializes /
    // variantSetNames) are explicitly excluded by the spec note in
    // §12.2.6 and should leave this false — they are processed by the
    // composition driver, not metadata resolution.
    // Trailing field so legacy 4-arg `{name, type, required, specForms}`
    // brace initializers continue to compile, defaulting to false.
    bool isListOp     = false;
};

// ============================================================
// FieldRegistry — enumerates valid fields per spec form
// ============================================================

class NANOUSD_CORE_API FieldRegistry {
public:
    static const FieldRegistry& Instance();

    // All descriptors for a given spec form
    const std::vector<const FieldDescriptor*>& GetFields(SpecType type) const;

    // Look up a descriptor by name (nullptr if unknown)
    const FieldDescriptor* Find(const Token& name) const;

    // Check whether a field name is valid for a given spec form
    bool IsValidField(const Token& name, SpecType type) const;

private:
    FieldRegistry();
    void Register(FieldDescriptor desc);

    std::unordered_map<std::string, FieldDescriptor> descriptors_;
    std::unordered_map<SpecType, std::vector<const FieldDescriptor*>> bySpecType_;
};

// ============================================================
// Fields — generic field storage (Token -> Value)
// ============================================================
// Per spec Section 7.2 the contents of a layer are values addressable via
// a (spec path, field name) pair.  Fields is the per-spec container.
//
// Storage is a flat `vector<pair<Token, Value>>`. Specs typically author
// 2-10 fields; with Tokens being interned (equality is a pointer compare),
// linear scan beats hashing in both time and memory at this scale. Switch
// rationale measured on a 250k-prim scene: ~600 MB saved vs. an
// unordered_map of equivalent contents (no per-node allocs, no separate
// bucket array, no fixed per-container overhead). Field order is not
// semantically meaningful — specs identify fields by name, not position.

class Fields {
public:
    Fields() = default;

    // --- Raw access ---

    bool Has(const Token& name) const {
        for (const auto& e : data_) if (e.first == name) return true;
        return false;
    }

    const Value* Get(const Token& name) const {
        for (const auto& e : data_) if (e.first == name) return &e.second;
        return nullptr;
    }

    void Set(const Token& name, Value v) {
        for (auto& e : data_) {
            if (e.first == name) { e.second = std::move(v); return; }
        }
        data_.emplace_back(name, std::move(v));
    }

    void Remove(const Token& name) {
        for (size_t i = 0; i < data_.size(); ++i) {
            if (data_[i].first == name) {
                if (i + 1 != data_.size()) data_[i] = std::move(data_.back());
                data_.pop_back();
                return;
            }
        }
    }

    // --- Typed getters (return std::optional) ---

    template <typename T>
    std::optional<T> GetAs(const Token& name) const {
        auto* v = Get(name);
        if (!v) return std::nullopt;
        auto* p = v->Get<T>();
        if (!p) return std::nullopt;
        return *p;
    }

    // --- Typed getter with fallback ---

    template <typename T>
    T GetOr(const Token& name, const T& fallback) const {
        auto opt = GetAs<T>(name);
        return opt.value_or(fallback);
    }

    // Iteration
    using VecType = std::vector<std::pair<Token, Value>>;
    VecType::const_iterator begin() const { return data_.begin(); }
    VecType::const_iterator end()   const { return data_.end(); }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

private:
    VecType data_;
};

// ============================================================
// Spec — a single spec with its type and generic field storage
// ============================================================
// Every spec is fundamentally a SpecType + Fields bag.  The typed accessor
// methods below provide convenience without losing the generic nature.

class NANOUSD_CORE_API Spec {
public:
    explicit Spec(SpecType type = SpecType::Prim) : type_(type) {}
    Spec(const Spec& o)
        : type_(o.type_),
          fields_(o.fields_),
          arcOpinionCache_(o.arcOpinionCache_.load(std::memory_order_relaxed)),
          backing_(o.backing_),
          deferredFieldsetIdx_(o.deferredFieldsetIdx_) {}
    Spec& operator=(const Spec& o) {
        if (this == &o) return *this;
        type_ = o.type_;
        fields_ = o.fields_;
        arcOpinionCache_.store(
            o.arcOpinionCache_.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        backing_ = o.backing_;
        deferredFieldsetIdx_ = o.deferredFieldsetIdx_;
        return *this;
    }
    Spec(Spec&& o) noexcept
        : type_(o.type_),
          fields_(std::move(o.fields_)),
          arcOpinionCache_(o.arcOpinionCache_.load(std::memory_order_relaxed)),
          backing_(o.backing_),
          deferredFieldsetIdx_(o.deferredFieldsetIdx_) {}
    Spec& operator=(Spec&& o) noexcept {
        if (this == &o) return *this;
        type_ = o.type_;
        fields_ = std::move(o.fields_);
        arcOpinionCache_.store(
            o.arcOpinionCache_.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        backing_ = o.backing_;
        deferredFieldsetIdx_ = o.deferredFieldsetIdx_;
        return *this;
    }

    SpecType GetType() const { return type_; }

    // --- Layer-owned deferral ---
    //
    // When a Spec is loaded from a format that supports lazy decode
    // (USDC), the parser sets up a deferral context: a raw pointer
    // to the Layer's `DecodeBacking` (kept stable for the Layer's
    // lifetime) and the fieldset index identifying this Spec's
    // authored fields in the source file.
    //
    // First access to any field via GetField/HasField asks the
    // backing to decode that single field by name and caches it in
    // `fields_`. Per-field laziness — querying defaultValue does
    // not pay typeName/custom/variability decode cost.
    //
    // USDA and in-memory specs never set this up; their fields are
    // populated eagerly at construction time.
    void SetDeferralContext(const class DecodeBacking* backing,
                             std::uint32_t fieldsetIdx) {
        backing_ = backing;
        deferredFieldsetIdx_ = fieldsetIdx;
        arcOpinionCache_.store(-1, std::memory_order_relaxed);
    }
    bool HasDeferredFields() const {
        return backing_ != nullptr && deferredFieldsetIdx_ != kNoFieldset;
    }
    static constexpr std::uint32_t kNoFieldset = ~std::uint32_t{0};

    // --- Generic field access ---
    //
    // GetFields() materialises every authored field via the Layer
    // backing (one round-trip into the crate file). Per-field
    // GetField/HasField only fault in the requested name.

    Fields& GetFields() {
        EnsureFullyDecoded();
        // Mutable Fields access can bypass SetField/RemoveField.
        arcOpinionCache_.store(-1, std::memory_order_relaxed);
        return fields_;
    }
    const Fields& GetFields() const {
        EnsureFullyDecoded();
        return fields_;
    }

    bool HasField(const Token& name) const {
        if (fields_.Has(name)) return true;
        if (HasDeferredFields()) {
            EnsureFieldDecoded(name);
            return fields_.Has(name);
        }
        return false;
    }
    const Value* GetField(const Token& name) const {
        if (auto* v = fields_.Get(name)) return v;
        if (HasDeferredFields()) {
            EnsureFieldDecoded(name);
            return fields_.Get(name);
        }
        return nullptr;
    }
    void SetField(const Token& name, Value v) {
        if (IsArcOpinionField(name))
            arcOpinionCache_.store(-1, std::memory_order_relaxed);
        fields_.Set(name, std::move(v));
    }
    void RemoveField(const Token& name) {
        if (IsArcOpinionField(name))
            arcOpinionCache_.store(-1, std::memory_order_relaxed);
        fields_.Remove(name);
    }

    bool HasArcOpinion() const {
        int cached = arcOpinionCache_.load(std::memory_order_relaxed);
        if (cached >= 0) return cached != 0;
        auto authored = [](const Value* v) { return v && !v->IsEmpty(); };
        bool has =
            authored(GetField(FieldNames::references)) ||
            authored(GetField(FieldNames::payload)) ||
            authored(GetField(FieldNames::variantSelection)) ||
            authored(GetField(FieldNames::inheritPaths)) ||
            authored(GetField(FieldNames::specializes));
        arcOpinionCache_.store(has ? 1 : 0, std::memory_order_relaxed);
        return has;
    }

    // Check field validity against the registry
    bool IsValidField(const Token& name) const {
        return FieldRegistry::Instance().IsValidField(name, type_);
    }

    // --- Convenience accessors for commonly used fields ---
    // These provide typed access with spec-defined fallback values.

    // Specifier (Prim, Variant)
    Specifier GetSpecifier() const;
    void SetSpecifier(Specifier s);

    // typeName (Prim, Variant, Attribute)
    Token GetTypeName() const;
    void SetTypeName(const Token& t);

    // active (Prim, Variant) — fallback true
    bool GetActive() const;
    void SetActive(bool v);

    // instanceable (Prim, Variant) — fallback false
    bool GetInstanceable() const;
    void SetInstanceable(bool v);

    // hidden (Prim, Variant, Attribute, Relationship) — fallback false
    bool GetHidden() const;
    void SetHidden(bool v);

    // custom (Attribute, Relationship) — fallback false
    bool GetCustom() const;
    void SetCustom(bool v);

    // variability (Attribute) — fallback Varying
    Variability GetVariability() const;
    void SetVariability(Variability v);

    // kind (Prim, Variant)
    Token GetKind() const;
    void SetKind(const Token& k);

    // documentation (Layer, Prim, Variant, Attribute, Relationship)
    String GetDocumentation() const;
    void SetDocumentation(const String& s);

    // comment (Layer, Prim, Variant, Attribute, Relationship)
    String GetComment() const;
    void SetComment(const String& s);

    // displayName (Prim, Variant, Attribute, Relationship)
    String GetDisplayName() const;
    void SetDisplayName(const String& s);

    // defaultPrim (Layer)
    Token GetDefaultPrim() const;
    void SetDefaultPrim(const Token& t);

    // Timing (Layer)
    double GetTimeCodesPerSecond() const;
    void SetTimeCodesPerSecond(double v);
    double GetFramesPerSecond() const;
    void SetFramesPerSecond(double v);
    double GetStartTimeCode() const;
    void SetStartTimeCode(double v);
    double GetEndTimeCode() const;
    void SetEndTimeCode(double v);

private:
    static bool IsArcOpinionField(const Token& name) {
        return name == FieldNames::references ||
               name == FieldNames::payload ||
               name == FieldNames::variantSelection ||
               name == FieldNames::inheritPaths ||
               name == FieldNames::specializes;
    }

    // Both helpers are out-of-line in spec.cpp / value.cpp because
    // they need the full DecodeBacking type, which would create a
    // header dependency cycle if pulled in here.
    void EnsureFullyDecoded() const;
    void EnsureFieldDecoded(const Token& name) const;

    SpecType type_;
    Fields fields_;
    mutable std::atomic<int> arcOpinionCache_{-1};
    const DecodeBacking* backing_ = nullptr;       // raw — Layer outlives Spec
    std::uint32_t deferredFieldsetIdx_ = kNoFieldset;
};

} // namespace nanousd
