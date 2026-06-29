// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "value.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace nanousd {

// ============================================================
// Schema property definition (from a schema)
// ============================================================

struct SchemaPropertyDef {
    std::string name;           // e.g. "points" or "collection:<__INSTANCE_NAME__>:expansionRule"
    std::string typeName;       // e.g. "float3[]", "token", "rel"
    Variability variability = Variability::Varying;
    std::optional<Value> fallback;
    bool hidden = false;
    std::string documentation;
};

// ============================================================
// Schema kind enumeration
// ============================================================

enum class SchemaKind { Typed, SingleApply, MultipleApply };

// ============================================================
// Schema definition — a parsed schema
// ============================================================

struct SchemaDef {
    std::string name;           // e.g. "Mesh", "CollectionAPI"
    SchemaKind kind;
    std::string parent;         // typed only: parent schema name
    bool isAbstract = false;
    std::string instancePrefix; // multipleApply only
    std::map<std::string, SchemaPropertyDef> properties;
    std::vector<std::string> builtIns;
    std::vector<std::string> autoApplies;
    std::map<std::string, SchemaPropertyDef> overrideProperties;
};

// ============================================================
// PrimDefinition — composed property set per Section 13.3.2.3
// ============================================================

struct NANOUSD_CORE_API PrimDefinition {
    std::vector<std::string> schemas;  // ordered: typed first, then applied
    std::map<std::string, SchemaPropertyDef> properties;  // composed property set
    // Cached schema-defined attribute names in the same map order as
    // `properties`, excluding relationship definitions (`typeName == "rel"`).
    // Rebuilt whenever a PrimDefinition is constructed by SchemaRegistry.
    std::vector<Token> attributeNameTokens;

    const SchemaPropertyDef* GetPropertyDef(const std::string& name) const;
    bool HasProperty(const std::string& name) const;
};

// ============================================================
// SchemaRegistry — singleton registry for schema definitions
// ============================================================

class NANOUSD_CORE_API SchemaRegistry {
public:
    static SchemaRegistry& GetInstance();

    // --- Registration ---
    bool RegisterSchema(SchemaDef def);
    bool LoadFromJSON(std::string_view json, std::string* error = nullptr);

    // --- Queries ---
    const SchemaDef* FindSchema(const std::string& name) const;
    bool IsTypedSchema(const std::string& name) const;
    bool IsConcreteTypedSchema(const std::string& name) const;
    bool IsAbstractTypedSchema(const std::string& name) const;
    bool IsAppliedSchema(const std::string& name) const;
    bool IsA(const std::string& typeName, const std::string& queryType) const;
    std::string ResolvePrimTypeName(
        const std::string& authoredTypeName,
        const Dictionary* fallbackPrimTypes = nullptr) const;
    std::vector<std::string> GetBuiltInAPISchemasForType(
        const std::string& typeName) const;
    std::vector<std::string> GetAutoAppliedAPISchemasForType(
        const std::string& typeName) const;

    // --- Prim definition building (spec Section 13.3.2.3) ---
    // Typed schema only (no applied schemas folded in).
    const PrimDefinition* GetPrimDefinition(const std::string& typeName) const;

    // Full prim definition: typed schema + applied API schemas.
    // This is the complete Section 13.3.2.3 algorithm.
    PrimDefinition BuildFullPrimDefinition(
        const std::string& typeName,
        const std::vector<std::string>& apiSchemas) const;

    // --- Reset (for testing) ---
    void Clear();

    // Monotonically increasing counter, bumped on each schema registration.
    // CompositionGraph uses this to lazily invalidate its prim def cache.
    uint64_t GetGeneration() const;

private:
    SchemaRegistry();

    void RegisterCoreSchemas();

    PrimDefinition BuildPrimDefinition(const std::string& schemaName) const;
    void ApplySchemaProperties(PrimDefinition& def,
                               const SchemaDef& schema) const;

    std::map<std::string, SchemaDef> schemas_;
    mutable std::map<std::string, PrimDefinition> primDefCache_;
    mutable std::mutex cacheMutex_;
    uint64_t generation_ = 0;
};

} // namespace nanousd
