// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "compose.h"
#include "schema.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nanousd {

// Forward declarations
class Stage;
class UsdPrim;
class UsdAttribute;
class UsdRelationship;

namespace detail {
size_t GetPrimAttributeCount(const UsdPrim& prim);
} // namespace detail

// --- Stage-level interpolation type per spec Section 12.5 ---

enum class UsdInterpolationType {
    Held,    // value is held to nearest previous authored sample
    Linear,  // basic linear interpolation between samples
};

// --- Value resolution result ---

struct UsdResolvedValue {
    Value value;
    bool found = false;  // true if a value was resolved (not a sentinel)
};

// --- Time sample resolution (spec Section 12.5) ---
// Resolves a time query against a Dictionary of time samples.
// Before first / after last authored sample: held.
// Between samples: uses the given interpolation type (Linear or Held).
UsdResolvedValue ResolveTimeSample(const Dictionary& samples, double time,
                                   UsdInterpolationType interp);

// ============================================================
// UsdObject — base for prims and properties (spec Section 10.2)
// ============================================================

class UsdObject {
public:
    // Defensive: UsdProperty / UsdAttribute / UsdRelationship / UsdPrim carry
    // non-trivial members (std::string, std::optional<Value>, owned Path etc.),
    // so deleting a derived instance through a UsdObject* or an owning
    // std::unique_ptr<UsdObject> without a virtual destructor would skip the
    // derived destructor and leak those members. Even though the current
    // codebase doesn't do polymorphic delete today, the inheritance shape
    // invites it — make the base destructor virtual so any such use is safe.
    virtual ~UsdObject() = default;

    const Path& GetPath() const { return path_; }
    const Token& GetName() const { return name_; }
    bool IsValid() const { return !path_.IsEmpty(); }

protected:
    UsdObject() = default;
    UsdObject(Path path, Token name)
        : path_(std::move(path)), name_(std::move(name)) {}

    Path path_;
    Token name_;
};

// ============================================================
// UsdProperty — base for attributes and relationships (spec Section 10.3)
// ============================================================

class NANOUSD_CORE_API UsdProperty : public UsdObject {
public:
    bool IsAuthored() const { return spec_ != nullptr; }
    bool IsCustom() const;
    bool IsHidden() const;
    String GetDisplayName() const;
    String GetDisplayGroup() const;
    String GetDocumentation() const;

protected:
    UsdProperty() = default;
    UsdProperty(Path path, Token name, const Spec* spec)
        : UsdObject(std::move(path), std::move(name)), spec_(spec) {}

    const Spec* spec_ = nullptr;
};

// ============================================================
// UsdAttribute — a composed attribute (spec Section 10.3.1)
// ============================================================

class NANOUSD_CORE_API UsdAttribute : public UsdProperty {
public:
    static constexpr size_t kNoLayer = ~size_t(0);

    UsdAttribute() = default;
    UsdAttribute(Path path, Token name, const Spec* spec)
        : UsdProperty(std::move(path), std::move(name), spec) {}
    UsdAttribute(Path path, Token name, const Spec* spec, size_t layerIndex)
        : UsdProperty(std::move(path), std::move(name), spec), layerIndex_(layerIndex) {}
    // Schema-only attribute (no authored spec in layer)
    UsdAttribute(Path path, Token name, SchemaPropertyDef schemaDef)
        : UsdProperty(std::move(path), std::move(name), nullptr),
          schemaDef_(std::move(schemaDef)), hasSchemaDef_(true) {}
    // Schema-only attribute with mutable layer (for write support)
    UsdAttribute(Path path, Token name, SchemaPropertyDef schemaDef, size_t layerIndex)
        : UsdProperty(std::move(path), std::move(name), nullptr),
          schemaDef_(std::move(schemaDef)), hasSchemaDef_(true), layerIndex_(layerIndex) {}
    // Authored attribute with schema fallback
    UsdAttribute(Path path, Token name, const Spec* spec, SchemaPropertyDef schemaDef)
        : UsdProperty(std::move(path), std::move(name), spec),
          schemaDef_(std::move(schemaDef)), hasSchemaDef_(true) {}
    // Authored attribute with schema fallback and mutable layer
    UsdAttribute(Path path, Token name, const Spec* spec,
                 SchemaPropertyDef schemaDef, size_t layerIndex)
        : UsdProperty(std::move(path), std::move(name), spec),
          schemaDef_(std::move(schemaDef)), hasSchemaDef_(true), layerIndex_(layerIndex) {}

    // Graph-aware constructors
    UsdAttribute(Path path, Token name, const Spec* spec,
                 CompositionGraph* graph, const PrimIndex* primIndex, size_t layerIndex)
        : UsdProperty(std::move(path), std::move(name), spec),
          graph_(graph), primIndex_(primIndex), layerIndex_(layerIndex) {}
    UsdAttribute(Path path, Token name, const Spec* spec,
                 SchemaPropertyDef schemaDef,
                 CompositionGraph* graph, const PrimIndex* primIndex, size_t layerIndex)
        : UsdProperty(std::move(path), std::move(name), spec),
          schemaDef_(std::move(schemaDef)), hasSchemaDef_(true),
          graph_(graph), primIndex_(primIndex), layerIndex_(layerIndex) {}
    UsdAttribute(Path path, Token name, SchemaPropertyDef schemaDef,
                 CompositionGraph* graph, const PrimIndex* primIndex, size_t layerIndex)
        : UsdProperty(std::move(path), std::move(name), nullptr),
          schemaDef_(std::move(schemaDef)), hasSchemaDef_(true),
          graph_(graph), primIndex_(primIndex), layerIndex_(layerIndex) {}

    Token GetTypeName() const;
    Variability GetVariability() const;

    // Value access — returns authored default, falling back to schema fallback.
    //
    // WARNING: When this falls back to GetFallback(), the returned pointer
    // points into this UsdAttribute's schemaDef_ member. If the UsdAttribute
    // is a temporary (e.g. returned by UsdPrim::GetAttribute()), the pointer
    // is dangling after the temporary is destroyed. Callers that store the
    // result must either copy the Value or ensure the UsdAttribute outlives
    // the pointer.
    const Value* GetDefault() const;
    bool HasDefault() const;

    // Schema fallback value (nullptr if no schema definition).
    // Same lifetime caveat as GetDefault() — pointer is into this object.
    const Value* GetFallback() const;

    // Full value resolution per spec Section 12.3.
    UsdResolvedValue Get(UsdTimeCode time = UsdTimeCode::Default()) const;
    UsdResolvedValue Get(UsdTimeCode time, UsdInterpolationType interp) const;

    // Whether this attribute is defined by a schema (vs purely custom/authored)
    bool IsSchemaDefined() const { return hasSchemaDef_; }
    bool HasAuthoredValue() const;
    bool HasAuthoredSpec() const { return spec_ != nullptr; }
    const Spec* GetAuthoredSpec() const { return spec_; }

    // TimeSamples
    bool HasTimeSamples() const;
    const Value* GetTimeSamplesField() const;
    std::vector<double> GetTimeSampleTimes() const;

    // Color-space metadata (spec Color chapter). GetColorSpace returns only
    // authored attribute colorSpace metadata; ComputeColorSpaceName applies
    // the attribute/prim/ancestor resolution rules.
    Token GetColorSpace() const;
    bool HasColorSpace() const;
    bool SetColorSpace(const Token& colorSpace);
    bool ClearColorSpace();
    Token ComputeColorSpaceName() const;

    // Connection targets
    bool HasConnections() const;
    // Spec §12.2.6: connectionPaths follow generic listop metadata
    // combining across the opinion stack. Returns the composed list
    // of connection target paths.
    std::vector<Path> GetConnections() const;

    // --- Write operations ---

    bool Set(Value val);
    bool SetTimeSample(double time, Value val);
    bool ClearDefault();
    bool ClearTimeSamples();
    bool Block();

private:
    Spec* GetOrCreateMutableSpec();
    Layer& GetMutableLayer() { return graph_->GetMutableLayer(layerIndex_); }
    const Layer& GetLayer() const { return graph_->GetLayer(layerIndex_); }

    SchemaPropertyDef schemaDef_;
    bool hasSchemaDef_ = false;
    CompositionGraph* graph_ = nullptr;
    const PrimIndex* primIndex_ = nullptr;
    size_t layerIndex_ = kNoLayer;
};

// ============================================================
// UsdRelationship — a composed relationship (spec Section 10.3.2)
// ============================================================

class NANOUSD_CORE_API UsdRelationship : public UsdProperty {
public:
    static constexpr size_t kNoLayer = ~size_t(0);

    UsdRelationship() = default;
    UsdRelationship(Path path, Token name, const Spec* spec)
        : UsdProperty(std::move(path), std::move(name), spec) {}
    UsdRelationship(Path path, Token name, const Spec* spec, size_t layerIndex)
        : UsdProperty(std::move(path), std::move(name), spec),
          layerIndex_(layerIndex) {}
    UsdRelationship(Path path, Token name, const Spec* spec,
                    CompositionGraph* graph, const PrimIndex* primIndex, size_t layerIndex)
        : UsdProperty(std::move(path), std::move(name), spec),
          graph_(graph), primIndex_(primIndex), layerIndex_(layerIndex) {}
    // Schema-only relationship (no authored spec, writable)
    UsdRelationship(Path path, Token name, size_t layerIndex)
        : UsdProperty(std::move(path), std::move(name), nullptr),
          layerIndex_(layerIndex) {}
    UsdRelationship(Path path, Token name,
                    CompositionGraph* graph, const PrimIndex* primIndex, size_t layerIndex)
        : UsdProperty(std::move(path), std::move(name), nullptr),
          graph_(graph), primIndex_(primIndex), layerIndex_(layerIndex) {}

    bool HasTargets() const;
    std::vector<Path> GetTargets() const;
    std::vector<Path> GetForwardedTargets() const;

    // --- Write operations ---

    bool SetTargets(const std::vector<Path>& targets);
    bool AddTarget(const Path& target);
    bool ClearTargets();

private:
    Spec* GetOrCreateMutableSpec();
    Layer& GetMutableLayer() { return graph_->GetMutableLayer(layerIndex_); }
    const Layer& GetLayer() const { return graph_->GetLayer(layerIndex_); }

    CompositionGraph* graph_ = nullptr;
    const PrimIndex* primIndex_ = nullptr;
    size_t layerIndex_ = kNoLayer;
};

// ============================================================
// UsdPrim — a composed prim on the stage (spec Section 10.2)
// ============================================================

class NANOUSD_CORE_API UsdPrim : public UsdObject {
public:
    UsdPrim() = default;

    // --- Metadata ---

    Specifier GetSpecifier() const;
    Token GetTypeName() const;
    bool IsActive() const;
    bool IsLoaded() const;
    bool IsModel() const;
    bool IsInstanceable() const;
    bool IsHidden() const;
    Token GetKind() const;
    String GetDocumentation() const;

    // Generic prim-level metadata accessor (spec §7.6 "Core metadata fields",
    // §12.2 "Metadata Resolution"). Composes the named field across the
    // opinion stack per the field's type:
    //   * Dictionary fields: combine per §6.6.2.1 / §12.2.5.
    //   * Other fields:      first authored opinion wins.
    // Asset path values are resolved against their authoring layer per §5.
    // Returns std::nullopt if no opinion authored the field anywhere on the
    // stack — callers that want a fallback value should use the typed
    // accessor for that field (IsActive, GetKind, GetDocumentation, etc.)
    // or apply the fallback themselves per §12.2.8.
    std::optional<Value> GetPrimMetadata(const Token& fieldName) const;

    // --- Specifier queries ---

    bool IsDefined() const;
    bool IsAbstract() const;

    // --- Children ---

    std::vector<UsdPrim> GetChildren() const;
    UsdPrim GetChild(const Token& name) const;
    bool HasChild(const Token& name) const;

    // --- Properties ---

    std::vector<Token> GetPropertyNames() const;
    // Same walk as GetPropertyNames, but filters to property names whose
    // authored (or schema-provided) spec is an attribute. Does the type
    // check once per unique name during the opinion-entry walk, avoiding
    // the O(entries × names) behavior of calling HasAttribute in a loop.
    std::vector<Token> GetAttributeNames() const;
    std::vector<Token> GetAuthoredAttributeNames() const;
    UsdAttribute GetAttribute(const Token& name) const;
    UsdRelationship GetRelationship(const Token& name) const;
    bool HasAttribute(const Token& name) const;
    bool HasRelationship(const Token& name) const;

    // --- Property ordering (spec Section 10.4) ---

    std::vector<Token> GetPropertyOrder() const;

    // --- Schema queries (spec Section 13) ---

    bool IsA(const Token& typeName) const;
    bool HasAPI(const Token& schemaName) const;
    bool HasAPI(const Token& schemaName, const Token& instanceName) const;
    std::vector<Token> GetAppliedSchemas() const;
    PrimDefinition GetPrimDefinition() const;
    Token ComputeColorSpaceName() const;

    // --- Collections (spec Section 15) ---
    //
    // Evaluates a CollectionAPI instance lazily from its composed includes /
    // excludes relationships. No membership is materialized during stage
    // population; callers pay only when querying a collection.
    std::vector<Path> ComputeCollectionMembership(const Token& instanceName) const;
    bool IsCollectionMember(const Token& instanceName, const Path& path) const;

    // --- Variants (spec Section 11.2) ---
    //
    // Reads compose ListOp<string> / Dictionary fields across the prim's
    // opinion stack. GetVariantNames scans source-layer Variant specs at
    // /primPath{setName=*} under each opinion entry and deduplicates —
    // variants declared across multiple layers for the same set are
    // unioned. Selection writes go to the layer at `layerIndex`, updating
    // the variantSelection Dictionary field on the prim spec; the caller
    // must invoke Stage::Recompose() for the new selection to take effect.

    std::vector<Token> GetVariantSetNames() const;
    std::vector<Token> GetVariantNames(const Token& setName) const;
    Token GetVariantSelection(const Token& setName) const;
    bool HasVariantSet(const Token& setName) const;
    bool SetVariantSelection(const Token& setName, const Token& variantName,
                             size_t layerIndex = 0);

    // --- Instancing (spec Section 11.3) ---

    bool IsInstance() const;
    bool IsPrototype() const;
    bool IsInPrototype() const;
    UsdPrim GetPrototype() const;
    std::vector<UsdPrim> GetInstances() const;

    static constexpr size_t kNoLayer = ~size_t(0);

    // Legacy constructors (single-layer mode)
    UsdPrim(Path path, Token name, const Spec* spec, size_t layerIndex)
        : UsdObject(std::move(path), std::move(name)), spec_(spec),
          layerIndex_(layerIndex) {}

    // Graph-aware constructor
    UsdPrim(Path path, Token name, const Spec* spec,
            CompositionGraph* graph, const PrimIndex* primIndex, size_t layerIndex)
        : UsdObject(std::move(path), std::move(name)), spec_(spec),
          graph_(graph), primIndex_(primIndex), layerIndex_(layerIndex) {}

    // Graph-aware constructor with Stage back-pointer (for instancing queries)
    UsdPrim(Path path, Token name, const Spec* spec,
            CompositionGraph* graph, const PrimIndex* primIndex, size_t layerIndex,
            const Stage* stage)
        : UsdObject(std::move(path), std::move(name)), spec_(spec),
          graph_(graph), primIndex_(primIndex), layerIndex_(layerIndex), stage_(stage) {}

    // --- Write operations ---

    UsdAttribute CreateAttribute(const Token& name, const Token& typeName,
                                 bool custom = true);

    UsdRelationship CreateRelationship(const Token& name);

    bool ApplyAPI(const Token& schemaName);
    bool ApplyAPI(const Token& schemaName, const Token& instanceName);

private:
    friend size_t detail::GetPrimAttributeCount(const UsdPrim& prim);

    Layer& GetMutableLayer() { return graph_->GetMutableLayer(layerIndex_); }
    const Layer& GetLayer() const { return graph_->GetLayer(layerIndex_); }
    void AppendAttributeNames(std::vector<Token>& names) const;

    const Spec* spec_ = nullptr;
    CompositionGraph* graph_ = nullptr;
    const PrimIndex* primIndex_ = nullptr;
    size_t layerIndex_ = kNoLayer;
    const Stage* stage_ = nullptr;
};

// ============================================================
// Stage — the composed scene (spec Section 10.1)
// ============================================================

class NANOUSD_CORE_API Stage {
public:
    static Stage Open(const std::string& filePath,
                      AssetResolver resolver = nullptr);
    // Open with a population mask. Mask paths populate only those prims and
    // their ancestors; an empty mask is equivalent to Open().
    static Stage OpenMasked(const std::string& filePath,
                            const std::vector<Path>& populationMask,
                            AssetResolver resolver = nullptr);

    static Stage CreateInMemory();

    static Stage CreateFromComposedLayer(Layer composedLayer);
    static Stage CreateFromComposedLayer(CompositionGraph graph);

    // --- Authoring ---

    UsdPrim DefinePrim(const Path& path, const Token& typeName = Token());

    // Re-run composition from the root layer, rebuilding the entire graph.
    // All existing UsdPrim handles become stale after this call.
    // Returns true on success, false on error (check GetError()).
    bool Recompose();

    // --- Queries ---

    bool IsValid() const { return valid_; }
    const std::string& GetError() const { return error_; }

    // Composition diagnostics — non-fatal issues collected during composition.
    // Any Warning or Error severity means the stage may not represent the scene correctly.
    const DiagnosticCollector& GetDiagnostics() const { return diagnostics_; }
    bool HasCompositionErrors() const { return diagnostics_.HasErrors(); }

    UsdPrim GetPseudoRoot() const;
    UsdPrim GetDefaultPrim() const;
    std::vector<UsdPrim> GetRootPrims() const;
    UsdPrim GetPrimAtPath(const Path& path) const;
    bool HasPrimAtPath(const Path& path) const;
    std::vector<UsdPrim> Traverse() const;

    // --- Stage metadata (from resolved root layer spec) ---

    double GetTimeCodesPerSecond() const;
    double GetFramesPerSecond() const;
    double GetStartTimeCode() const;
    double GetEndTimeCode() const;

    // --- Interpolation type (spec Section 12.5) ---

    UsdInterpolationType GetInterpolationType() const { return interpolationType_; }
    void SetInterpolationType(UsdInterpolationType t) { interpolationType_ = t; }

    // --- Access to composed layer spec (resolved root layer metadata) ---

    const Spec& GetComposedLayerSpec() const { return graph_.composedLayerSpec; }
    Spec& GetMutableComposedLayerSpec() { return graph_.composedLayerSpec; }

    // --- Access to the composition graph ---

    const CompositionGraph& GetGraph() const { return graph_; }
    CompositionGraph& GetGraph() { return graph_; }

    // Returns the resolved file path of the root layer ("" for in-memory stages)
    const std::string& GetRootLayerPath() const {
        return graph_.layerPaths.empty() ? s_empty : graph_.layerPaths[0];
    }

    // --- Backward compatibility: composed layer facade ---
    // These return the root layer for write operations, and use the graph
    // for read operations. This preserves compatibility for code that
    // previously worked with a single flat layer.

    const Layer& GetComposedLayer() const { return graph_.GetRootLayer(); }
    Layer& GetMutableLayer() { return graph_.GetRootLayer(); }

    // --- Instancing queries (spec Section 11.3) ---

    bool IsInstance(const Path& path) const;
    bool IsPrototypeRoot(const Path& path) const;
    Path GetPrototypePath(const Path& instancePath) const;
    const std::vector<Path>& GetPrototypePaths() const;
    const std::vector<Path>& GetInstancePaths(const Path& prototypePath) const;
    void EnsurePrototypeInstanceIndex() const;

private:
    friend class UsdPrim;

    struct PopulatedPrimRecord {
        Path path;
        const PrimIndex* primIndex = nullptr;
        size_t strongestLayerIndex = 0;
        // Empty when the strongest spec is in identity-mapped namespace.
        Path strongestSourcePath;
    };

    Stage() = default;

    static Stage OpenImpl(const std::string& filePath,
                          AssetResolver resolver,
                          std::vector<Path> populationMask,
                          bool hasPopulationMask);
    void Populate();
    void EnsurePopulated() const;
    bool ShouldPopulate(const Spec& spec) const;
    bool IsInPrototypeNamespace(const Path& path) const;
    bool PopulationMaskIncludes(const Path& path) const;
    void BuildPrototypeSubtree(const Path& protoPath, const Path& instancePath);
    // Recursive implementation: origInstancePath + origProtoPath are carried
    // unchanged so the per-entry arc-origin filter can compare against the
    // instance root and arc targets can be remapped to prototype namespace.
    // See BuildFilteredPrimIndex in stage.cpp.
    void BuildPrototypeSubtreeImpl(const Path& protoPath, const Path& instancePath,
                                    const Path& origInstancePath,
                                    const Path& origProtoPath);

    CompositionGraph graph_;
    bool valid_ = false;
    mutable bool populateDirty_ = true;
    std::string error_;
    DiagnosticCollector diagnostics_;
    UsdInterpolationType interpolationType_ = UsdInterpolationType::Linear;
    bool hasPopulationMask_ = false;
    std::vector<Path> populationMask_;

    // Populated prims (depth-first order), with enough provenance to
    // construct traversal handles without rescanning the opinion stack.
    mutable std::vector<PopulatedPrimRecord> populatedPrims_;

    // Instancing state (spec Section 11.3)
    std::vector<Path> prototypePaths_;                                            // /__Prototype_0, ...
    PathMap<Path> instanceToPrototype_;              // instance -> prototype root
    mutable PathMap<std::vector<Path>> prototypeToInstances_; // prototype -> instances
    mutable bool prototypeToInstancesDirty_ = true;
    PathSet prototypeRoots_;                         // fast lookup

    static const std::string s_empty;
    static const std::vector<Path> s_emptyPaths;
};

// Flatten a Stage into a single Layer with no external dependencies.
// Walks all composed prims and properties, resolving sampled attribute values
// through the opinion stack. Composition operators and value-source metadata
// such as clips/splines are removed from the result.
NANOUSD_CORE_API Layer FlattenStage(const Stage& stage);

} // namespace nanousd
