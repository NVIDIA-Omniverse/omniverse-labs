// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/decode_backing.h"
#include "nanousd/layer.h"

namespace nanousd {

// --- Spec deferral helpers ---
// Defined here (out-of-line from spec.h) so they can use the full
// DecodeBacking type without forcing a header dependency cycle.

void Spec::EnsureFullyDecoded() const {
    if (!HasDeferredFields()) return;
    auto* mutSelf = const_cast<Spec*>(this);
    const DecodeBacking* backing = backing_;
    std::uint32_t idx = deferredFieldsetIdx_;
    // Clear deferral first so re-entrant queries during the decode
    // don't loop. The backing is responsible for filling fields_.
    mutSelf->backing_ = nullptr;
    mutSelf->deferredFieldsetIdx_ = kNoFieldset;
    backing->DecodeAllFields(idx, *mutSelf);
}

void Spec::EnsureFieldDecoded(const Token& name) const {
    // Per-field decode would corrupt callers that hold a `Value*` from
    // a prior GetField across this call: Fields' vector storage moves
    // entries on insert, invalidating earlier pointers. PostProcessSubLayers
    // (subLayers + subLayerOffsets) and any consumer that reads two fields
    // back-to-back hit this. Decode the whole fieldset in one shot — same
    // end state, plus pointers stay stable because subsequent GetField
    // calls find their entry in `fields_` without growing it.
    (void)name;
    EnsureFullyDecoded();
}

// ============================================================
// FieldRegistry
// ============================================================

void FieldRegistry::Register(FieldDescriptor desc) {
    std::string key = desc.name.GetString();
    auto& stored = descriptors_[key];
    stored = std::move(desc);
    for (auto st : stored.specForms) {
        bySpecType_[st].push_back(&descriptors_[key]);
    }
}

const FieldRegistry& FieldRegistry::Instance() {
    static FieldRegistry reg;
    return reg;
}

const std::vector<const FieldDescriptor*>& FieldRegistry::GetFields(SpecType type) const {
    static const std::vector<const FieldDescriptor*> empty;
    auto it = bySpecType_.find(type);
    return it != bySpecType_.end() ? it->second : empty;
}

const FieldDescriptor* FieldRegistry::Find(const Token& name) const {
    auto it = descriptors_.find(name.GetString());
    return it != descriptors_.end() ? &it->second : nullptr;
}

bool FieldRegistry::IsValidField(const Token& name, SpecType type) const {
    auto* desc = Find(name);
    if (!desc) return false;
    return desc->specForms.count(type) > 0;
}

// Helper to build a spec-form set
static std::unordered_set<SpecType> Forms(std::initializer_list<SpecType> list) {
    return {list};
}

// All spec forms that behave like prims (Prim + Variant per spec 7.6.7)
static const auto PrimLike = Forms({SpecType::Prim, SpecType::Variant});
static const auto Properties = Forms({SpecType::Attribute, SpecType::Relationship});
static const auto AllSpecs = Forms({
    SpecType::Layer, SpecType::Prim, SpecType::Attribute,
    SpecType::Relationship, SpecType::VariantSet, SpecType::Variant
});

FieldRegistry::FieldRegistry() {
    using namespace FieldNames;
    using S = SpecType;

    // --- Hierarchy fields ---
    Register({primChildren,       TypeId::Unknown, false, {S::Layer, S::Prim, S::Variant}});
    Register({propertyChildren,   TypeId::Unknown, false, {S::Prim, S::Variant}});
    Register({variantSetChildren, TypeId::Unknown, false, {S::Prim, S::Variant}});
    Register({variantChildren,    TypeId::Unknown, false, {S::VariantSet}});

    // --- Composition fields ---
    Register({subLayers,         TypeId::Unknown, false, {S::Layer}});
    Register({subLayerOffsets,   TypeId::Unknown, false, {S::Layer}});
    Register({defaultPrim,       TypeId::Token,   false, {S::Layer}});
    Register({layerRelocates,    TypeId::Unknown, false, {S::Layer}});
    Register({references,        TypeId::Unknown, false, {S::Prim, S::Variant}});
    Register({payload,           TypeId::Unknown, false, {S::Prim, S::Variant}});
    Register({inheritPaths,      TypeId::Unknown, false, {S::Prim, S::Variant}});
    Register({specializes,       TypeId::Unknown, false, {S::Prim, S::Variant}});
    Register({variantSetNames,   TypeId::Unknown, false, {S::Prim, S::Variant}});
    Register({variantSelection,  TypeId::Unknown, false, {S::Prim, S::Variant}});

    // --- Population fields ---
    Register({specifier,         TypeId::Unknown, true,  {S::Prim, S::Variant}});
    Register({typeName,          TypeId::Token,   true,  {S::Prim, S::Variant, S::Attribute}});
    Register({active,            TypeId::Bool,    false, {S::Prim, S::Variant}});
    Register({instanceable,      TypeId::Bool,    false, {S::Prim, S::Variant}});
    Register({kind,              TypeId::Token,   false, {S::Prim, S::Variant}});
    Register({primOrder,         TypeId::Unknown, false, {S::Layer, S::Prim, S::Variant}});
    Register({propertyOrder,     TypeId::Unknown, false, {S::Prim, S::Variant}});

    // --- Value fields ---
    Register({custom,            TypeId::Bool,    true,  {S::Attribute, S::Relationship}});
    Register({variability,       TypeId::Unknown, true,  {S::Attribute}});
    Register({defaultValue,      TypeId::Unknown, false, {S::Attribute}});
    Register({timeSamples,       TypeId::Unknown, false, {S::Attribute}});
    // connectionPaths / targetPaths are listops (spec §12.2.6 — they
    // follow the generic listop combining rules in value resolution).
    Register({connectionPaths,   TypeId::Unknown, false, {S::Attribute},   /*isListOp=*/true});
    Register({spline,            TypeId::Spline,  false, {S::Attribute}});
    Register({allowedTokens,     TypeId::Unknown, false, {S::Attribute}});
    Register({colorSpace,        TypeId::Token,   false, {S::Attribute}});
    Register({targetPaths,       TypeId::Unknown, false, {S::Relationship},/*isListOp=*/true});

    // --- Timing fields (Layer) ---
    Register({timeCodesPerSecond,TypeId::Double,  false, {S::Layer}});
    Register({framesPerSecond,   TypeId::Double,  false, {S::Layer}});
    Register({startTimeCode,     TypeId::Double,  false, {S::Layer}});
    Register({endTimeCode,       TypeId::Double,  false, {S::Layer}});

    // --- User interface fields ---
    Register({displayName,       TypeId::String,  false, {S::Prim, S::Variant, S::Attribute, S::Relationship}});
    Register({displayGroup,      TypeId::String,  false, {S::Attribute, S::Relationship}});
    Register({displayGroupOrder, TypeId::Unknown, false, {S::Prim, S::Variant}});
    Register({hidden,            TypeId::Bool,    false, {S::Prim, S::Variant, S::Attribute, S::Relationship}});
    Register({documentation,     TypeId::String,  false, {S::Layer, S::Prim, S::Variant, S::Attribute, S::Relationship}});

    // --- User fields ---
    Register({customData,        TypeId::Dictionary, false, {S::Prim, S::Variant, S::Attribute, S::Relationship}});
    Register({customLayerData,   TypeId::Dictionary, false, {S::Layer}});
    Register({assetInfo,         TypeId::Dictionary, false, {S::Prim, S::Variant, S::Attribute, S::Relationship}});
    Register({comment,           TypeId::String,     false, {S::Layer, S::Prim, S::Variant, S::Attribute, S::Relationship}});

    // --- Schema fields ---
    // apiSchemas is a generic prim-level listop metadata field (spec
    // §13.2.1.2 + §12.2.6). isListOp=true so GetPrimMetadata combines
    // it across opinions per §6.6.3.6.
    Register({apiSchemas,        TypeId::Unknown, false, {S::Prim, S::Variant}, /*isListOp=*/true});
    Register({fallbackPrimTypes,  TypeId::Dictionary, false, {S::Layer}});

    // --- Value clip fields (spec §12.3.4) ---
    // clips: prim-level Dictionary keyed by clip-set name; each inner
    // dict carries per-set metadata (assetPaths, active, times,
    // primPath, manifestAssetPath, template*). Rides the §12.2.5
    // dict-combining path (TypeId::Dictionary triggers that in
    // GetPrimMetadata).
    // clipSets: ListOp<string> naming the search order of clip sets;
    // rides §12.2.6 listop combining.
    Register({clips,             TypeId::Dictionary, false, {S::Prim, S::Variant}});
    Register({clipSets,          TypeId::Unknown,    false, {S::Prim, S::Variant}, /*isListOp=*/true});
}

// ============================================================
// Spec convenience accessors
// ============================================================

// Convenience accessors all route through GetField() so that the
// lazy field initializer fires on first access. With deferral
// applied universally to USDC-loaded specs, the dedicated members
// (specifier_, typeName_, …) on the Spec aren't usable until the
// init lambda has run — going through GetField is the single
// correct path and the per-call cost is one extra branch + map
// lookup vs. the pre-deferral fields_.Get directly.

Specifier Spec::GetSpecifier() const {
    if (auto* v = GetField(FieldNames::specifier)) {
        if (auto* p = v->Get<Int>()) return static_cast<Specifier>(*p);
    }
    return Specifier::Over;
}

void Spec::SetSpecifier(Specifier s) {
    fields_.Set(FieldNames::specifier, Value(static_cast<Int>(s)));
}

Token Spec::GetTypeName() const {
    if (auto* v = GetField(FieldNames::typeName)) {
        if (auto* t = v->Get<Token>()) return *t;
    }
    return Token("");
}

void Spec::SetTypeName(const Token& t) {
    fields_.Set(FieldNames::typeName, Value(t));
}

bool Spec::GetActive() const {
    if (auto* v = GetField(FieldNames::active)) {
        if (auto* b = v->Get<Bool>()) return *b;
    }
    return true;
}

void Spec::SetActive(bool v) {
    fields_.Set(FieldNames::active, Value(v));
}

bool Spec::GetInstanceable() const {
    if (auto* v = GetField(FieldNames::instanceable)) {
        if (auto* b = v->Get<Bool>()) return *b;
    }
    return false;
}

void Spec::SetInstanceable(bool v) {
    fields_.Set(FieldNames::instanceable, Value(v));
}

bool Spec::GetHidden() const {
    if (auto* v = GetField(FieldNames::hidden)) {
        if (auto* b = v->Get<Bool>()) return *b;
    }
    return false;
}

void Spec::SetHidden(bool v) {
    fields_.Set(FieldNames::hidden, Value(v));
}

bool Spec::GetCustom() const {
    if (auto* v = GetField(FieldNames::custom)) {
        if (auto* b = v->Get<Bool>()) return *b;
    }
    return false;
}

void Spec::SetCustom(bool v) {
    fields_.Set(FieldNames::custom, Value(v));
}

Variability Spec::GetVariability() const {
    if (auto* v = GetField(FieldNames::variability)) {
        if (auto* p = v->Get<Int>()) return static_cast<Variability>(*p);
    }
    return Variability::Varying;
}

void Spec::SetVariability(Variability v) {
    fields_.Set(FieldNames::variability, Value(static_cast<Int>(v)));
}

Token Spec::GetKind() const {
    if (auto* v = GetField(FieldNames::kind)) {
        if (auto* t = v->Get<Token>()) return *t;
    }
    return Token("");
}

void Spec::SetKind(const Token& k) {
    fields_.Set(FieldNames::kind, Value(k));
}

String Spec::GetDocumentation() const {
    if (auto* v = GetField(FieldNames::documentation)) {
        if (auto* s = v->Get<String>()) return *s;
    }
    return "";
}

void Spec::SetDocumentation(const String& s) {
    fields_.Set(FieldNames::documentation, Value(s));
}

String Spec::GetComment() const {
    if (auto* v = GetField(FieldNames::comment)) {
        if (auto* s = v->Get<String>()) return *s;
    }
    return "";
}

void Spec::SetComment(const String& s) {
    fields_.Set(FieldNames::comment, Value(s));
}

String Spec::GetDisplayName() const {
    if (auto* v = GetField(FieldNames::displayName)) {
        if (auto* s = v->Get<String>()) return *s;
    }
    return "";
}

void Spec::SetDisplayName(const String& s) {
    fields_.Set(FieldNames::displayName, Value(s));
}

Token Spec::GetDefaultPrim() const {
    if (auto* v = GetField(FieldNames::defaultPrim)) {
        if (auto* t = v->Get<Token>()) return *t;
    }
    return Token("");
}

void Spec::SetDefaultPrim(const Token& t) {
    fields_.Set(FieldNames::defaultPrim, Value(t));
}

double Spec::GetTimeCodesPerSecond() const {
    if (auto* v = GetField(FieldNames::timeCodesPerSecond)) {
        if (auto* d = v->Get<Double>()) return *d;
    }
    return 24.0;
}

void Spec::SetTimeCodesPerSecond(double v) {
    fields_.Set(FieldNames::timeCodesPerSecond, Value(v));
}

double Spec::GetFramesPerSecond() const {
    if (auto* v = GetField(FieldNames::framesPerSecond)) {
        if (auto* d = v->Get<Double>()) return *d;
    }
    return 24.0;
}

void Spec::SetFramesPerSecond(double v) {
    fields_.Set(FieldNames::framesPerSecond, Value(v));
}

double Spec::GetStartTimeCode() const {
    if (auto* v = GetField(FieldNames::startTimeCode)) {
        if (auto* d = v->Get<Double>()) return *d;
    }
    return 0.0;
}

void Spec::SetStartTimeCode(double v) {
    fields_.Set(FieldNames::startTimeCode, Value(v));
}

double Spec::GetEndTimeCode() const {
    if (auto* v = GetField(FieldNames::endTimeCode)) {
        if (auto* d = v->Get<Double>()) return *d;
    }
    return 0.0;
}

void Spec::SetEndTimeCode(double v) {
    fields_.Set(FieldNames::endTimeCode, Value(v));
}

// ============================================================
// Layer implementation
// ============================================================

const Spec* Layer::GetSpec(const std::string& pathText) const {
    return GetSpec(Path::Parse(pathText));
}

Spec* Layer::GetSpec(const std::string& pathText) {
    return GetSpec(Path::Parse(pathText));
}

const Spec* Layer::GetSpecOfType(const Path& path, SpecType type) const {
    auto* spec = GetSpec(path);
    if (spec && spec->GetType() == type) return spec;
    return nullptr;
}

Spec* Layer::GetMutableSpecOfType(const Path& path, SpecType type) {
    auto* spec = GetSpec(path);
    if (spec && spec->GetType() == type) return spec;
    return nullptr;
}

const Spec* Layer::GetPrimSpec(const Path& path) const {
    return GetSpecOfType(path, SpecType::Prim);
}

const Spec* Layer::GetAttributeSpec(const Path& path) const {
    return GetSpecOfType(path, SpecType::Attribute);
}

const Spec* Layer::GetRelationshipSpec(const Path& path) const {
    return GetSpecOfType(path, SpecType::Relationship);
}

Spec* Layer::GetMutablePrimSpec(const Path& path) {
    return GetMutableSpecOfType(path, SpecType::Prim);
}

Spec* Layer::GetMutableAttributeSpec(const Path& path) {
    return GetMutableSpecOfType(path, SpecType::Attribute);
}

Spec* Layer::GetMutableRelationshipSpec(const Path& path) {
    return GetMutableSpecOfType(path, SpecType::Relationship);
}

// Spec §6.6.2.1 — dictionary combining rule.
Dictionary CombineDicts(const Dictionary& stronger, const Dictionary& weaker) {
    Dictionary result = stronger;
    for (const auto& [key, weakVal] : weaker) {
        auto it = result.find(key);
        if (it == result.end()) {
            // Rule 4: key only in weaker → take weaker's value.
            result.emplace(key, weakVal);
            continue;
        }
        // Key is in both. If both values are dicts, recurse (rule 3);
        // otherwise the stronger value is kept (rule 2, already in
        // `result`).
        const auto* strongDict = it->second.Get<Dictionary>();
        const auto* weakDict   = weakVal.Get<Dictionary>();
        if (strongDict && weakDict) {
            it->second = Value(CombineDicts(*strongDict, *weakDict));
        }
    }
    return result;
}

} // namespace nanousd
