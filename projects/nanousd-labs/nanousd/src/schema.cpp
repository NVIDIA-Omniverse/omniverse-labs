// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/schema.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace nanousd {

// ============================================================
// PrimDefinition
// ============================================================

const SchemaPropertyDef* PrimDefinition::GetPropertyDef(const std::string& name) const {
    auto it = properties.find(name);
    return it != properties.end() ? &it->second : nullptr;
}

bool PrimDefinition::HasProperty(const std::string& name) const {
    return properties.count(name) > 0;
}

namespace {
void RefreshPrimDefinitionCaches(PrimDefinition& def) {
    def.attributeNameTokens.clear();
    def.attributeNameTokens.reserve(def.properties.size());
    for (const auto& [name, prop] : def.properties) {
        if (prop.typeName == "rel") continue;
        def.attributeNameTokens.emplace_back(name);
    }
}

std::string AppliedSchemaBaseName(const std::string& apiEntry) {
    auto colonPos = apiEntry.find(':');
    return colonPos == std::string::npos ? apiEntry
                                         : apiEntry.substr(0, colonPos);
}

void AppendUnique(std::vector<std::string>& values, const std::string& value) {
    if (std::find(values.begin(), values.end(), value) == values.end())
        values.push_back(value);
}

void AppendFallbackPrimTypeCandidates(const Value& value,
                                      std::vector<std::string>& out) {
    if (const auto* tokens = value.Get<std::vector<Token>>()) {
        for (const auto& token : *tokens) out.push_back(token.GetString());
        return;
    }
    if (const auto* strings = value.Get<std::vector<std::string>>()) {
        out.insert(out.end(), strings->begin(), strings->end());
        return;
    }
    if (const auto* values = value.Get<std::vector<Value>>()) {
        for (const auto& item : *values) {
            if (const auto* token = item.Get<Token>()) {
                out.push_back(token->GetString());
            } else if (const auto* str = item.Get<std::string>()) {
                out.push_back(*str);
            }
        }
        return;
    }
    if (const auto* token = value.Get<Token>()) {
        out.push_back(token->GetString());
    } else if (const auto* str = value.Get<std::string>()) {
        out.push_back(*str);
    }
}
} // namespace

// ============================================================
// Minimal JSON parser — handles objects, arrays, strings,
// numbers, bools, and null. ~200 lines.
// ============================================================

namespace {

struct JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

enum class JsonType { Null, Bool, Number, String, Array, Object };

struct JsonValue {
    JsonType type = JsonType::Null;
    bool boolVal = false;
    double numVal = 0.0;
    std::string strVal;
    JsonArray arrVal;
    JsonObject objVal;

    bool IsNull() const { return type == JsonType::Null; }
    bool IsBool() const { return type == JsonType::Bool; }
    bool IsNumber() const { return type == JsonType::Number; }
    bool IsString() const { return type == JsonType::String; }
    bool IsArray() const { return type == JsonType::Array; }
    bool IsObject() const { return type == JsonType::Object; }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view input) : input_(input), pos_(0) {}

    JsonValue Parse() {
        SkipWS();
        auto val = ParseValue();
        SkipWS();
        return val;
    }

    bool HasError() const { return !error_.empty(); }
    const std::string& GetError() const { return error_; }

private:
    std::string_view input_;
    size_t pos_;
    std::string error_;

    char Peek() const { return pos_ < input_.size() ? input_[pos_] : '\0'; }
    char Next() { return pos_ < input_.size() ? input_[pos_++] : '\0'; }

    void SkipWS() {
        while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_])))
            ++pos_;
    }

    bool Expect(char c) {
        SkipWS();
        if (Peek() == c) { ++pos_; return true; }
        error_ = std::string("Expected '") + c + "' at position " + std::to_string(pos_);
        return false;
    }

    JsonValue ParseValue() {
        SkipWS();
        char c = Peek();
        if (c == '"') return ParseString();
        if (c == '{') return ParseObject();
        if (c == '[') return ParseArray();
        if (c == 't' || c == 'f') return ParseBool();
        if (c == 'n') return ParseNull();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return ParseNumber();
        error_ = std::string("Unexpected character '") + c + "' at position " + std::to_string(pos_);
        return {};
    }

    JsonValue ParseString() {
        JsonValue v;
        v.type = JsonType::String;
        ++pos_; // skip opening "
        while (pos_ < input_.size() && input_[pos_] != '"') {
            if (input_[pos_] == '\\') {
                ++pos_;
                if (pos_ >= input_.size()) break;
                char esc = input_[pos_];
                switch (esc) {
                    case '"': v.strVal += '"'; break;
                    case '\\': v.strVal += '\\'; break;
                    case '/': v.strVal += '/'; break;
                    case 'n': v.strVal += '\n'; break;
                    case 't': v.strVal += '\t'; break;
                    case 'r': v.strVal += '\r'; break;
                    default: v.strVal += esc; break;
                }
            } else {
                v.strVal += input_[pos_];
            }
            ++pos_;
        }
        if (pos_ < input_.size()) ++pos_; // skip closing "
        return v;
    }

    JsonValue ParseNumber() {
        JsonValue v;
        v.type = JsonType::Number;
        size_t start = pos_;
        if (Peek() == '-') ++pos_;
        while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) ++pos_;
        if (pos_ < input_.size() && input_[pos_] == '.') {
            ++pos_;
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) ++pos_;
        }
        if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) ++pos_;
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) ++pos_;
        }
        std::string numStr(input_.substr(start, pos_ - start));
        v.numVal = std::stod(numStr);
        return v;
    }

    JsonValue ParseBool() {
        JsonValue v;
        v.type = JsonType::Bool;
        if (input_.substr(pos_, 4) == "true") { v.boolVal = true; pos_ += 4; }
        else if (input_.substr(pos_, 5) == "false") { v.boolVal = false; pos_ += 5; }
        else { error_ = "Invalid boolean at position " + std::to_string(pos_); }
        return v;
    }

    JsonValue ParseNull() {
        JsonValue v;
        if (input_.substr(pos_, 4) == "null") { pos_ += 4; }
        else { error_ = "Invalid null at position " + std::to_string(pos_); }
        return v;
    }

    JsonValue ParseArray() {
        JsonValue v;
        v.type = JsonType::Array;
        ++pos_; // skip [
        SkipWS();
        if (Peek() == ']') { ++pos_; return v; }
        while (!HasError()) {
            v.arrVal.push_back(ParseValue());
            if (HasError()) break;
            SkipWS();
            if (Peek() == ',') { ++pos_; SkipWS(); continue; }
            break;
        }
        if (!Expect(']')) return v;
        return v;
    }

    JsonValue ParseObject() {
        JsonValue v;
        v.type = JsonType::Object;
        ++pos_; // skip {
        SkipWS();
        if (Peek() == '}') { ++pos_; return v; }
        while (!HasError()) {
            SkipWS();
            if (Peek() != '"') { error_ = "Expected string key at " + std::to_string(pos_); break; }
            auto key = ParseString();
            if (HasError()) break;
            SkipWS();
            if (!Expect(':')) break;
            SkipWS();
            auto val = ParseValue();
            if (HasError()) break;
            v.objVal[key.strVal] = std::move(val);
            SkipWS();
            if (Peek() == ',') { ++pos_; SkipWS(); continue; }
            break;
        }
        if (!HasError()) Expect('}');
        return v;
    }
};

// Convert a JSON fallback value to a Value based on the type name
Value JsonFallbackToValue(const JsonValue& jv, const std::string& typeName) {
    if (typeName == "bool" && jv.IsBool()) {
        return Value(jv.boolVal);
    }
    if (typeName == "int" && jv.IsNumber()) {
        return Value(Int(static_cast<int>(jv.numVal)));
    }
    if (typeName == "float" && jv.IsNumber()) {
        return Value(Float(static_cast<float>(jv.numVal)));
    }
    // Float special values: "inf", "-inf" as string fallbacks
    if (typeName == "float" && jv.IsString()) {
        if (jv.strVal == "inf") return Value(Float(std::numeric_limits<float>::infinity()));
        if (jv.strVal == "-inf") return Value(Float(-std::numeric_limits<float>::infinity()));
    }
    if (typeName == "double" && jv.IsNumber()) {
        return Value(Double(jv.numVal));
    }
    if (typeName == "string" && jv.IsString()) {
        return Value(jv.strVal);
    }
    if (typeName == "token" && jv.IsString()) {
        return Value(Token(jv.strVal));
    }
    if ((typeName == "float2" || typeName == "texCoord2f") &&
        jv.IsArray() && jv.arrVal.size() == 2) {
        GfVec2f v;
        for (size_t i = 0; i < 2; ++i)
            v[i] = jv.arrVal[i].IsNumber() ? Float(static_cast<float>(jv.arrVal[i].numVal)) : Float(0);
        return Value(v);
    }
    // Vec3 types: vector3f, point3f, float3, normal3f, color3f
    if ((typeName == "vector3f" || typeName == "point3f" || typeName == "float3" ||
         typeName == "normal3f" || typeName == "color3f") && jv.IsArray() && jv.arrVal.size() == 3) {
        GfVec3f v;
        for (size_t i = 0; i < 3; ++i)
            v[i] = jv.arrVal[i].IsNumber() ? Float(static_cast<float>(jv.arrVal[i].numVal)) : Float(0);
        return Value(v);
    }
    // Quaternion types: quatf — JSON array is [w, i, j, k], internal storage is [i, j, k, r]
    if (typeName == "quatf" && jv.IsArray() && jv.arrVal.size() == 4) {
        float w = jv.arrVal[0].IsNumber() ? static_cast<float>(jv.arrVal[0].numVal) : 0.0f;
        float i = jv.arrVal[1].IsNumber() ? static_cast<float>(jv.arrVal[1].numVal) : 0.0f;
        float j = jv.arrVal[2].IsNumber() ? static_cast<float>(jv.arrVal[2].numVal) : 0.0f;
        float k = jv.arrVal[3].IsNumber() ? static_cast<float>(jv.arrVal[3].numVal) : 0.0f;
        GfQuatf q;
        q[0] = i; q[1] = j; q[2] = k; q[3] = w;  // i, j, k, r
        return Value(q);
    }
    // For other types, store as string representation
    if (jv.IsString()) {
        return Value(jv.strVal);
    }
    if (jv.IsNumber()) {
        return Value(Double(jv.numVal));
    }
    if (jv.IsBool()) {
        return Value(jv.boolVal);
    }
    return Value();
}

SchemaPropertyDef ParsePropertyDef(const std::string& name, const JsonValue& jv) {
    SchemaPropertyDef prop;
    prop.name = name;

    if (jv.IsObject()) {
        auto typeIt = jv.objVal.find("type");
        if (typeIt != jv.objVal.end() && typeIt->second.IsString())
            prop.typeName = typeIt->second.strVal;

        auto varIt = jv.objVal.find("variability");
        if (varIt != jv.objVal.end() && varIt->second.IsString()) {
            prop.variability = (varIt->second.strVal == "uniform")
                ? Variability::Uniform : Variability::Varying;
        }

        auto fallIt = jv.objVal.find("fallback");
        if (fallIt != jv.objVal.end() && !fallIt->second.IsNull()) {
            prop.fallback = JsonFallbackToValue(fallIt->second, prop.typeName);
        }

        auto hiddenIt = jv.objVal.find("hidden");
        if (hiddenIt != jv.objVal.end() && hiddenIt->second.IsBool())
            prop.hidden = hiddenIt->second.boolVal;

        auto docIt = jv.objVal.find("documentation");
        if (docIt != jv.objVal.end() && docIt->second.IsString())
            prop.documentation = docIt->second.strVal;
    }

    return prop;
}

} // anonymous namespace

// ============================================================
// SchemaRegistry
// ============================================================

SchemaRegistry& SchemaRegistry::GetInstance() {
    static SchemaRegistry instance;
    return instance;
}

SchemaRegistry::SchemaRegistry() {
    RegisterCoreSchemas();
}

void SchemaRegistry::Clear() {
    // Clear cache under lock, then re-register (which also locks internally)
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        primDefCache_.clear();
    }
    schemas_.clear();
    ++generation_;
    RegisterCoreSchemas();
}

uint64_t SchemaRegistry::GetGeneration() const {
    return generation_;
}

bool SchemaRegistry::RegisterSchema(SchemaDef def) {
    if (def.name.empty()) return false;
    if (schemas_.count(def.name) > 0) return false;
    std::string name = def.name;
    schemas_.emplace(name, std::move(def));
    // Invalidate any cached prim definitions that might be affected
    std::lock_guard<std::mutex> lock(cacheMutex_);
    primDefCache_.clear();
    return true;
}

bool SchemaRegistry::LoadFromJSON(std::string_view json, std::string* error) {
    JsonParser parser(json);
    auto root = parser.Parse();

    if (parser.HasError()) {
        if (error) *error = parser.GetError();
        return false;
    }

    if (!root.IsObject()) {
        if (error) *error = "Root must be a JSON object";
        return false;
    }

    // Find the "schemas" key
    auto schemasIt = root.objVal.find("schemas");
    if (schemasIt == root.objVal.end() || !schemasIt->second.IsObject()) {
        if (error) *error = "Missing or invalid 'schemas' object";
        return false;
    }

    for (const auto& [schemaName, schemaObj] : schemasIt->second.objVal) {
        if (!schemaObj.IsObject()) continue;

        SchemaDef def;
        def.name = schemaName;

        // schemaKind
        auto kindIt = schemaObj.objVal.find("schemaKind");
        if (kindIt != schemaObj.objVal.end() && kindIt->second.IsString()) {
            const auto& k = kindIt->second.strVal;
            if (k == "typed") def.kind = SchemaKind::Typed;
            else if (k == "singleApply") def.kind = SchemaKind::SingleApply;
            else if (k == "multipleApply") def.kind = SchemaKind::MultipleApply;
            else {
                if (error) *error = "Unknown schemaKind '" + k + "' for " + schemaName;
                return false;
            }
        } else {
            if (error) *error = "Missing schemaKind for " + schemaName;
            return false;
        }

        // parent (typed only)
        auto parentIt = schemaObj.objVal.find("parent");
        if (parentIt != schemaObj.objVal.end() && parentIt->second.IsString())
            def.parent = parentIt->second.strVal;

        // abstract
        auto absIt = schemaObj.objVal.find("abstract");
        if (absIt != schemaObj.objVal.end() && absIt->second.IsBool())
            def.isAbstract = absIt->second.boolVal;

        // instancePrefix (multipleApply)
        auto ipIt = schemaObj.objVal.find("instancePrefix");
        if (ipIt != schemaObj.objVal.end() && ipIt->second.IsString())
            def.instancePrefix = ipIt->second.strVal;

        // properties
        auto propsIt = schemaObj.objVal.find("properties");
        if (propsIt != schemaObj.objVal.end() && propsIt->second.IsObject()) {
            for (const auto& [propName, propObj] : propsIt->second.objVal) {
                def.properties[propName] = ParsePropertyDef(propName, propObj);
            }
        }

        // builtIns
        auto biIt = schemaObj.objVal.find("builtIns");
        if (biIt != schemaObj.objVal.end() && biIt->second.IsArray()) {
            for (const auto& item : biIt->second.arrVal) {
                if (item.IsString()) def.builtIns.push_back(item.strVal);
            }
        }

        // autoApplies
        auto aaIt = schemaObj.objVal.find("autoApplies");
        if (aaIt != schemaObj.objVal.end() && aaIt->second.IsArray()) {
            for (const auto& item : aaIt->second.arrVal) {
                if (item.IsString()) def.autoApplies.push_back(item.strVal);
            }
        }

        // overrideProperties
        auto opIt = schemaObj.objVal.find("overrideProperties");
        if (opIt != schemaObj.objVal.end() && opIt->second.IsObject()) {
            for (const auto& [propName, propObj] : opIt->second.objVal) {
                def.overrideProperties[propName] = ParsePropertyDef(propName, propObj);
            }
        }

        if (!RegisterSchema(std::move(def))) {
            if (error) *error = "Failed to register schema '" + schemaName + "' (duplicate?)";
            return false;
        }
    }

    // Invalidate prim def cache and bump generation so CompositionGraphs
    // lazily rebuild their cached PrimDefinitions.
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        primDefCache_.clear();
    }
    ++generation_;

    return true;
}

const SchemaDef* SchemaRegistry::FindSchema(const std::string& name) const {
    auto it = schemas_.find(name);
    return it != schemas_.end() ? &it->second : nullptr;
}

bool SchemaRegistry::IsTypedSchema(const std::string& name) const {
    auto* s = FindSchema(name);
    return s && s->kind == SchemaKind::Typed;
}

bool SchemaRegistry::IsConcreteTypedSchema(const std::string& name) const {
    auto* s = FindSchema(name);
    return s && s->kind == SchemaKind::Typed && !s->isAbstract;
}

bool SchemaRegistry::IsAbstractTypedSchema(const std::string& name) const {
    auto* s = FindSchema(name);
    return s && s->kind == SchemaKind::Typed && s->isAbstract;
}

bool SchemaRegistry::IsAppliedSchema(const std::string& name) const {
    auto* s = FindSchema(name);
    return s && (s->kind == SchemaKind::SingleApply || s->kind == SchemaKind::MultipleApply);
}

bool SchemaRegistry::IsA(const std::string& typeName, const std::string& queryType) const {
    if (typeName == queryType) return true;

    // Walk inheritance chain
    std::string current = typeName;
    while (true) {
        auto* schema = FindSchema(current);
        if (!schema || schema->kind != SchemaKind::Typed) return false;
        if (schema->parent.empty()) return false;
        if (schema->parent == queryType) return true;
        current = schema->parent;
    }
}

std::string SchemaRegistry::ResolvePrimTypeName(
        const std::string& authoredTypeName,
        const Dictionary* fallbackPrimTypes) const {
    if (authoredTypeName.empty()) return {};
    if (IsConcreteTypedSchema(authoredTypeName)) return authoredTypeName;
    if (IsAbstractTypedSchema(authoredTypeName)) return {};
    if (!fallbackPrimTypes) return {};

    auto it = fallbackPrimTypes->find(authoredTypeName);
    if (it == fallbackPrimTypes->end()) return {};

    std::vector<std::string> candidates;
    AppendFallbackPrimTypeCandidates(it->second, candidates);
    for (const auto& candidate : candidates) {
        if (IsConcreteTypedSchema(candidate)) return candidate;
    }
    return {};
}

std::vector<std::string> SchemaRegistry::GetBuiltInAPISchemasForType(
        const std::string& typeName) const {
    std::vector<std::string> result;
    auto* primDef = GetPrimDefinition(typeName);
    if (!primDef) return result;

    for (const auto& schemaName : primDef->schemas) {
        if (IsAppliedSchema(AppliedSchemaBaseName(schemaName))) {
            AppendUnique(result, schemaName);
        }
    }
    return result;
}

std::vector<std::string> SchemaRegistry::GetAutoAppliedAPISchemasForType(
        const std::string& typeName) const {
    std::vector<std::string> result;
    if (!IsConcreteTypedSchema(typeName)) return result;

    for (const auto& [name, schema] : schemas_) {
        if (schema.kind != SchemaKind::SingleApply) continue;
        for (const auto& target : schema.autoApplies) {
            if (typeName == target || IsA(typeName, target)) {
                AppendUnique(result, name);
                break;
            }
        }
    }
    return result;
}

const PrimDefinition* SchemaRegistry::GetPrimDefinition(const std::string& typeName) const {
    std::lock_guard<std::mutex> lock(cacheMutex_);

    // Check cache
    auto cacheIt = primDefCache_.find(typeName);
    if (cacheIt != primDefCache_.end()) return &cacheIt->second;

    // Build and cache
    auto* schema = FindSchema(typeName);
    if (!schema || schema->kind != SchemaKind::Typed || schema->isAbstract) return nullptr;

    auto def = BuildPrimDefinition(typeName);
    auto [it, _] = primDefCache_.emplace(typeName, std::move(def));
    return &it->second;
}

PrimDefinition SchemaRegistry::BuildPrimDefinition(const std::string& schemaName) const {
    PrimDefinition def;

    // Step 1: Collect the typed schema inheritance chain (strongest first)
    std::vector<const SchemaDef*> chain;
    std::string current = schemaName;
    while (!current.empty()) {
        auto* schema = FindSchema(current);
        if (!schema || schema->kind != SchemaKind::Typed) break;
        chain.push_back(schema);
        current = schema->parent;
    }

    // Step 2: Apply properties from weakest to strongest (reverse order)
    // so that stronger schemas override weaker ones
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        ApplySchemaProperties(def, **it);
        def.schemas.insert(def.schemas.begin(), (*it)->name);
    }

    // Step 3: Apply built-in API schemas from the primary schema
    // Handles multi-apply instance names (e.g. "CollectionAPI:colliders")
    if (!chain.empty()) {
        for (const auto& apiEntry : chain.front()->builtIns) {
            std::string schemaName = apiEntry;
            std::string instanceName;
            auto colonPos = apiEntry.find(':');
            if (colonPos != std::string::npos) {
                schemaName = apiEntry.substr(0, colonPos);
                instanceName = apiEntry.substr(colonPos + 1);
            }
            auto* apiSchema = FindSchema(schemaName);
            if (apiSchema && (apiSchema->kind == SchemaKind::SingleApply ||
                              apiSchema->kind == SchemaKind::MultipleApply)) {
                // For multi-apply, substitute instance name into property templates
                for (const auto& [templateName, prop] : apiSchema->properties) {
                    std::string propName = templateName;
                    if (!instanceName.empty()) {
                        const std::string placeholder = "<__INSTANCE_NAME__>";
                        auto pos = propName.find(placeholder);
                        if (pos != std::string::npos) {
                            propName.replace(pos, placeholder.size(), instanceName);
                        }
                    }
                    if (def.properties.count(propName) == 0) {
                        SchemaPropertyDef instProp = prop;
                        instProp.name = propName;
                        def.properties[propName] = std::move(instProp);
                    }
                }
                def.schemas.push_back(apiEntry);
            }
        }
    }

    // Step 4: Apply overrideProperties from the primary schema
    if (!chain.empty() && !chain.front()->overrideProperties.empty()) {
        for (const auto& [propName, overrideProp] : chain.front()->overrideProperties) {
            auto it = def.properties.find(propName);
            if (it != def.properties.end()) {
                // Override only the fallback if it's set
                if (overrideProp.fallback) {
                    it->second.fallback = overrideProp.fallback;
                }
            }
        }
    }

    RefreshPrimDefinitionCaches(def);
    return def;
}

void SchemaRegistry::ApplySchemaProperties(PrimDefinition& def,
                                           const SchemaDef& schema) const {
    for (const auto& [name, prop] : schema.properties) {
        // Overwrite — caller applies weakest first, so later (stronger) wins
        def.properties[name] = prop;
    }
}

PrimDefinition SchemaRegistry::BuildFullPrimDefinition(
        const std::string& typeName,
        const std::vector<std::string>& apiSchemas) const {

    // Start with the typed schema's prim definition
    PrimDefinition def;
    if (!typeName.empty()) {
        auto* baseDef = GetPrimDefinition(typeName);
        if (baseDef) {
            def = *baseDef;
        }
    }

    // Apply each applied API schema's properties.
    // For multi-apply schemas like "CollectionAPI:lights", substitute the
    // instance name into property name templates containing <__INSTANCE_NAME__>.
    for (const auto& apiEntry : apiSchemas) {
        std::string schemaName = apiEntry;
        std::string instanceName;

        auto colonPos = apiEntry.find(':');
        if (colonPos != std::string::npos) {
            schemaName = apiEntry.substr(0, colonPos);
            instanceName = apiEntry.substr(colonPos + 1);
        }

        auto* apiSchema = FindSchema(schemaName);
        if (!apiSchema) continue;
        if (apiSchema->kind != SchemaKind::SingleApply &&
            apiSchema->kind != SchemaKind::MultipleApply) continue;

        // Add to schema list
        def.schemas.push_back(apiEntry);

        for (const auto& [templateName, prop] : apiSchema->properties) {
            std::string propName = templateName;
            // Substitute <__INSTANCE_NAME__> for multi-apply
            if (!instanceName.empty()) {
                const std::string placeholder = "<__INSTANCE_NAME__>";
                auto pos = propName.find(placeholder);
                if (pos != std::string::npos) {
                    propName.replace(pos, placeholder.size(), instanceName);
                }
            }
            // API schema properties don't override typed schema properties
            if (def.properties.count(propName) == 0) {
                SchemaPropertyDef instProp = prop;
                instProp.name = propName;
                def.properties[propName] = std::move(instProp);
            }
        }
    }

    RefreshPrimDefinitionCaches(def);
    return def;
}

// ============================================================
// Core schema definitions
// ============================================================

void SchemaRegistry::RegisterCoreSchemas() {
    // ---------------------------------------------------------------
    // Normative schemas from AOUSD USD Core Specification v1.0.1
    // Sections 14 (Color) and 15 (Collections)
    // ---------------------------------------------------------------
    static const char* kNormativeSchemaJSON = R"({
  "schemas": {
    "CollectionAPI": {
      "schemaKind": "multipleApply",
      "instancePrefix": "collection",
      "properties": {
        "collection:<__INSTANCE_NAME__>:expansionRule": {
          "type": "token",
          "variability": "uniform",
          "fallback": "expandPrims",
          "documentation": "How included paths are expanded: explicitOnly, expandPrims, or expandPrimsAndProperties"
        },
        "collection:<__INSTANCE_NAME__>:includeRoot": {
          "type": "bool",
          "variability": "uniform",
          "fallback": false,
          "documentation": "Whether the collection includes the absolute root path"
        },
        "collection:<__INSTANCE_NAME__>:includes": {
          "type": "rel",
          "documentation": "Relationship targets included in this collection"
        },
        "collection:<__INSTANCE_NAME__>:excludes": {
          "type": "rel",
          "documentation": "Relationship targets excluded from this collection"
        },
        "collection:<__INSTANCE_NAME__>": {
          "type": "opaque",
          "documentation": "Property that can be referenced to include another collection"
        }
      }
    },
    "ColorSpaceAPI": {
      "schemaKind": "singleApply",
      "properties": {
        "colorSpace:name": {
          "type": "token",
          "variability": "uniform",
          "fallback": "",
          "documentation": "Color space applicable to attributes on this prim and descendants"
        }
      }
    },
    "ColorSpaceDefinitionAPI": {
      "schemaKind": "multipleApply",
      "instancePrefix": "colorSpaceDefinition",
      "properties": {
        "colorSpaceDefinition:<__INSTANCE_NAME__>:name": {
          "type": "token",
          "variability": "uniform",
          "fallback": "custom",
          "documentation": "Name of the custom color space"
        },
        "colorSpaceDefinition:<__INSTANCE_NAME__>:redChroma": {
          "type": "float2",
          "variability": "varying",
          "fallback": [1.0, 0.0],
          "documentation": "Red chromaticity coordinate"
        },
        "colorSpaceDefinition:<__INSTANCE_NAME__>:greenChroma": {
          "type": "float2",
          "variability": "varying",
          "fallback": [0.0, 1.0],
          "documentation": "Green chromaticity coordinate"
        },
        "colorSpaceDefinition:<__INSTANCE_NAME__>:blueChroma": {
          "type": "float2",
          "variability": "varying",
          "fallback": [0.0, 0.0],
          "documentation": "Blue chromaticity coordinate"
        },
        "colorSpaceDefinition:<__INSTANCE_NAME__>:whitePoint": {
          "type": "float2",
          "variability": "varying",
          "fallback": [0.33333333, 0.33333333],
          "documentation": "Whitepoint chromaticity coordinate"
        },
        "colorSpaceDefinition:<__INSTANCE_NAME__>:gamma": {
          "type": "float",
          "variability": "varying",
          "fallback": 1.0,
          "documentation": "Gamma value of the log section"
        },
        "colorSpaceDefinition:<__INSTANCE_NAME__>:linearBias": {
          "type": "float",
          "variability": "varying",
          "fallback": 0.0,
          "documentation": "Linear bias of the log section"
        }
      }
    }
  }
})";

    std::string err;
    LoadFromJSON(kNormativeSchemaJSON, &err);

    // ---------------------------------------------------------------
    // Geometry schemas inferred from AOUSD geometry-wg specifications
    // ---------------------------------------------------------------
    static const char* kGeometrySchemaJSON = R"JSON({
  "schemas": {
    "Imageable": {
      "schemaKind": "typed",
      "parent": "",
      "abstract": true,
      "properties": {
        "visibility": {
          "type": "token",
          "variability": "varying",
          "fallback": "inherited",
          "documentation": "Pruning visibility: inherited or invisible"
        },
        "purpose": {
          "type": "token",
          "variability": "uniform",
          "fallback": "default",
          "documentation": "Imaging purpose: default, guide, proxy, or render"
        },
        "proxyPrim": {
          "type": "rel",
          "documentation": "Optional link from render subgraph to proxy subgraph"
        }
      }
    },
    "GeomModelAPI": {
      "schemaKind": "singleApply",
      "properties": {
        "model:drawMode": {
          "type": "token",
          "variability": "uniform",
          "fallback": "inherited",
          "documentation": "Alternate imaging: origin, bounds, cards, default, inherited"
        },
        "model:applyDrawMode": {
          "type": "bool",
          "variability": "uniform",
          "fallback": false,
          "documentation": "Whether this prim's drawMode is applied even when not a model root"
        },
        "model:drawModeColor": {
          "type": "float3",
          "fallback": [0.18, 0.18, 0.18],
          "documentation": "Fallback color for bounds/origin/cards draw modes"
        },
        "model:cardGeometry": {
          "type": "token",
          "variability": "uniform",
          "fallback": "cross",
          "documentation": "Card geometry for cards draw mode: cross, box, fromTexture"
        },
        "model:cardTextureXPos": { "type": "asset", "documentation": "+X card texture" },
        "model:cardTextureYPos": { "type": "asset", "documentation": "+Y card texture" },
        "model:cardTextureZPos": { "type": "asset", "documentation": "+Z card texture" },
        "model:cardTextureXNeg": { "type": "asset", "documentation": "-X card texture" },
        "model:cardTextureYNeg": { "type": "asset", "documentation": "-Y card texture" },
        "model:cardTextureZNeg": { "type": "asset", "documentation": "-Z card texture" }
      }
    },
    "Scope": {
      "schemaKind": "typed",
      "parent": "Imageable",
      "properties": {}
    },
    "Xformable": {
      "schemaKind": "typed",
      "parent": "Imageable",
      "abstract": true,
      "properties": {
        "xformOpOrder": {
          "type": "token[]",
          "variability": "uniform",
          "documentation": "Ordered list of xform operations"
        }
      }
    },
    "Xform": {
      "schemaKind": "typed",
      "parent": "Xformable",
      "properties": {}
    },
    "Boundable": {
      "schemaKind": "typed",
      "parent": "Xformable",
      "abstract": true,
      "properties": {
        "extent": {
          "type": "float3[]",
          "variability": "varying",
          "documentation": "Axis-aligned bounding box as two float3 values (min, max)"
        }
      }
    },
    "Gprim": {
      "schemaKind": "typed",
      "parent": "Boundable",
      "abstract": true,
      "properties": {
        "orientation": {
          "type": "token",
          "variability": "uniform",
          "fallback": "rightHanded",
          "documentation": "Surface winding orientation: rightHanded or leftHanded"
        },
        "doubleSided": {
          "type": "bool",
          "variability": "uniform",
          "fallback": false,
          "documentation": "Whether the surface is visible from both sides"
        },
        "primvars:displayColor": {
          "type": "color3f[]",
          "variability": "varying",
          "documentation": "Albedo color for visualization"
        },
        "primvars:displayOpacity": {
          "type": "float[]",
          "variability": "varying",
          "documentation": "Opacity for visualization (0=transparent, 1=opaque)"
        }
      }
    },
    "PointBased": {
      "schemaKind": "typed",
      "parent": "Gprim",
      "abstract": true,
      "properties": {
        "points": {
          "type": "point3f[]",
          "variability": "varying",
          "documentation": "Primary point positions"
        },
        "normals": {
          "type": "normal3f[]",
          "variability": "varying",
          "documentation": "Surface normals"
        },
        "velocities": {
          "type": "vector3f[]",
          "variability": "varying",
          "documentation": "Per-point velocities for motion blur"
        },
        "accelerations": {
          "type": "vector3f[]",
          "variability": "varying",
          "documentation": "Per-point accelerations for motion blur"
        }
      }
    },
    "Mesh": {
      "schemaKind": "typed",
      "parent": "PointBased",
      "properties": {
        "faceVertexCounts": {
          "type": "int[]",
          "variability": "varying",
          "documentation": "Number of vertices per face"
        },
        "faceVertexIndices": {
          "type": "int[]",
          "variability": "varying",
          "documentation": "Indices into points array for each face vertex"
        },
        "subdivisionScheme": {
          "type": "token",
          "variability": "uniform",
          "fallback": "catmullClark",
          "documentation": "Subdivision scheme: catmullClark, loop, bilinear, or none"
        },
        "interpolateBoundary": {
          "type": "token",
          "variability": "uniform",
          "fallback": "edgeAndCorner",
          "documentation": "Boundary interpolation rule"
        },
        "faceVaryingLinearInterpolation": {
          "type": "token",
          "variability": "uniform",
          "fallback": "cornersPlus1",
          "documentation": "Face-varying interpolation rule"
        },
        "holeIndices": {
          "type": "int[]",
          "variability": "varying",
          "documentation": "Indices of faces to be treated as holes"
        },
        "cornerIndices": {
          "type": "int[]",
          "variability": "varying",
          "documentation": "Vertex indices for subdivision corners"
        },
        "cornerSharpnesses": {
          "type": "float[]",
          "variability": "varying",
          "documentation": "Sharpness values for subdivision corners"
        },
        "creaseIndices": {
          "type": "int[]",
          "variability": "varying",
          "documentation": "Vertex indices for subdivision creases"
        },
        "creaseLengths": {
          "type": "int[]",
          "variability": "varying",
          "documentation": "Number of vertices per crease"
        },
        "creaseSharpnesses": {
          "type": "float[]",
          "variability": "varying",
          "documentation": "Sharpness values for subdivision creases"
        }
      }
    },
    "Points": {
      "schemaKind": "typed",
      "parent": "PointBased",
      "properties": {
        "widths": {
          "type": "float[]",
          "variability": "varying",
          "documentation": "Diameter of each sphere or disk"
        },
        "ids": {
          "type": "int64[]",
          "variability": "varying",
          "documentation": "Optional stable particle ids for topologically varying points"
        }
      }
    },
    "Curves": {
      "schemaKind": "typed",
      "parent": "PointBased",
      "abstract": true,
      "properties": {
        "curveVertexCounts": {
          "type": "int[]",
          "variability": "varying",
          "documentation": "Number of vertices per curve"
        },
        "widths": {
          "type": "float[]",
          "variability": "varying",
          "documentation": "Width of each curve segment"
        }
      }
    },
    "BasisCurves": {
      "schemaKind": "typed",
      "parent": "Curves",
      "properties": {
        "type": {
          "type": "token",
          "variability": "uniform",
          "fallback": "cubic",
          "documentation": "Curve type: linear or cubic"
        },
        "basis": {
          "type": "token",
          "variability": "uniform",
          "fallback": "bezier",
          "documentation": "Cubic basis: bezier, bspline, or catmullRom"
        },
        "wrap": {
          "type": "token",
          "variability": "uniform",
          "fallback": "nonperiodic",
          "documentation": "Wrap mode: nonperiodic, periodic, or pinned"
        }
      }
    },
    "Cube": {
      "schemaKind": "typed",
      "parent": "Gprim",
      "properties": {
        "size": {
          "type": "double",
          "variability": "varying",
          "fallback": 2.0,
          "documentation": "Side length of the cube"
        }
      }
    },
    "Sphere": {
      "schemaKind": "typed",
      "parent": "Gprim",
      "properties": {
        "radius": {
          "type": "double",
          "variability": "varying",
          "fallback": 1.0,
          "documentation": "Radius of the sphere"
        }
      }
    },
    "Cone": {
      "schemaKind": "typed",
      "parent": "Gprim",
      "properties": {
        "height": {
          "type": "double",
          "variability": "varying",
          "fallback": 2.0,
          "documentation": "Length of the cone spine along the axis"
        },
        "radius": {
          "type": "double",
          "variability": "varying",
          "fallback": 1.0,
          "documentation": "Radius of the cone base"
        },
        "axis": {
          "type": "token",
          "variability": "uniform",
          "fallback": "Z",
          "documentation": "Axis along which the cone spine is aligned: X, Y, or Z"
        }
      }
    },
    "Cylinder": {
      "schemaKind": "typed",
      "parent": "Gprim",
      "properties": {
        "height": {
          "type": "double",
          "variability": "varying",
          "fallback": 2.0,
          "documentation": "Length of the cylinder spine along the axis"
        },
        "radius": {
          "type": "double",
          "variability": "varying",
          "fallback": 1.0,
          "documentation": "Radius of the cylinder"
        },
        "axis": {
          "type": "token",
          "variability": "uniform",
          "fallback": "Z",
          "documentation": "Axis along which the cylinder spine is aligned: X, Y, or Z"
        }
      }
    },
    "Cylinder_1": {
      "schemaKind": "typed",
      "parent": "Gprim",
      "properties": {
        "height": {
          "type": "double",
          "variability": "varying",
          "fallback": 2.0,
          "documentation": "Length of the cylinder spine along the axis"
        },
        "radiusTop": {
          "type": "double",
          "variability": "varying",
          "fallback": 1.0,
          "documentation": "Radius of the top of the cylinder (positive axis end)"
        },
        "radiusBottom": {
          "type": "double",
          "variability": "varying",
          "fallback": 1.0,
          "documentation": "Radius of the bottom of the cylinder (negative axis end)"
        },
        "axis": {
          "type": "token",
          "variability": "uniform",
          "fallback": "Z",
          "documentation": "Axis along which the cylinder spine is aligned: X, Y, or Z"
        }
      }
    },
    "Capsule": {
      "schemaKind": "typed",
      "parent": "Gprim",
      "properties": {
        "height": {
          "type": "double",
          "variability": "varying",
          "fallback": 1.0,
          "documentation": "Length of the capsule cylindrical section along the axis"
        },
        "radius": {
          "type": "double",
          "variability": "varying",
          "fallback": 0.5,
          "documentation": "Radius of the capsule hemispheres and cylindrical section"
        },
        "axis": {
          "type": "token",
          "variability": "uniform",
          "fallback": "Z",
          "documentation": "Axis along which the capsule spine is aligned: X, Y, or Z"
        }
      }
    },
    "Capsule_1": {
      "schemaKind": "typed",
      "parent": "Gprim",
      "properties": {
        "height": {
          "type": "double",
          "variability": "varying",
          "fallback": 1.0,
          "documentation": "Length of the capsule cylindrical section along the axis"
        },
        "radiusTop": {
          "type": "double",
          "variability": "varying",
          "fallback": 0.5,
          "documentation": "Radius of the top hemisphere (positive axis end)"
        },
        "radiusBottom": {
          "type": "double",
          "variability": "varying",
          "fallback": 0.5,
          "documentation": "Radius of the bottom hemisphere (negative axis end)"
        },
        "axis": {
          "type": "token",
          "variability": "uniform",
          "fallback": "Z",
          "documentation": "Axis along which the capsule spine is aligned: X, Y, or Z"
        }
      }
    },
    "Plane": {
      "schemaKind": "typed",
      "parent": "Gprim",
      "properties": {
        "width": {
          "type": "double",
          "variability": "varying",
          "fallback": 2.0,
          "documentation": "Width of the plane along the width direction"
        },
        "length": {
          "type": "double",
          "variability": "varying",
          "fallback": 2.0,
          "documentation": "Length of the plane along the length direction"
        },
        "axis": {
          "type": "token",
          "variability": "uniform",
          "fallback": "Z",
          "documentation": "Axis the surface normal aligns with: X, Y, or Z"
        }
      }
    },
    "GeomSubset": {
      "schemaKind": "typed",
      "parent": "",
      "properties": {
        "elementType": {
          "type": "token",
          "variability": "uniform",
          "fallback": "face",
          "documentation": "Type of geometric element the indices reference: face, point, edge, segment, or tetrahedron"
        },
        "indices": {
          "type": "int[]",
          "variability": "varying",
          "documentation": "Set of indices identifying elements in this subset"
        },
        "familyName": {
          "type": "token",
          "variability": "uniform",
          "fallback": "",
          "documentation": "Name of the family of subsets to which this subset belongs"
        }
      }
    }
  }
})JSON";

    LoadFromJSON(kGeometrySchemaJSON, &err);

    // Material schemas (from OpenUSD UsdShade domain)
    // Kept separate from core and geometry schemas
    static const char* kMaterialSchemaJSON = R"JSON({
  "schemas": {
    "NodeGraph": {
      "schemaKind": "typed",
      "parent": "",
      "properties": {}
    },
    "Material": {
      "schemaKind": "typed",
      "parent": "NodeGraph",
      "properties": {
        "outputs:surface": {
          "type": "token",
          "variability": "uniform",
          "documentation": "Terminal output representing the surface shader connection"
        },
        "outputs:displacement": {
          "type": "token",
          "variability": "uniform",
          "documentation": "Terminal output representing the displacement shader connection"
        },
        "outputs:volume": {
          "type": "token",
          "variability": "uniform",
          "documentation": "Terminal output representing the volume shader connection"
        }
      }
    },
    "MaterialBindingAPI": {
      "schemaKind": "singleApply",
      "properties": {
        "material:binding": {
          "type": "rel",
          "variability": "uniform",
          "documentation": "Direct binding to a Material prim"
        }
      }
    }
  }
})JSON";

    LoadFromJSON(kMaterialSchemaJSON, &err);

    // Physics schemas (from AOUSD physics-wg specification)
    // Kept separate from core, geometry, and material schemas
    static const char* kPhysicsSchemaJSON = R"JSON({
  "schemas": {
    "PhysicsScene": {
      "schemaKind": "typed",
      "parent": "",
      "properties": {
        "physics:gravityDirection": {
          "type": "vector3f",
          "variability": "varying",
          "fallback": [0, 0, 0],
          "documentation": "Direction of gravitational acceleration in world space. Zero vector requests negative stage upAxis."
        },
        "physics:gravityMagnitude": {
          "type": "float",
          "variability": "varying",
          "fallback": "-inf",
          "documentation": "Magnitude of gravitational acceleration. Negative values request Earth gravity. Units: distance/second/second."
        }
      }
    },
    "PhysicsCollisionGroup": {
      "schemaKind": "typed",
      "parent": "",
      "builtIns": ["CollectionAPI:colliders"],
      "properties": {
        "physics:filteredGroups": {
          "type": "rel",
          "variability": "uniform",
          "documentation": "Multi-target relationship to groups with which members must not collide."
        },
        "physics:invertFilteredGroups": {
          "type": "bool",
          "variability": "varying",
          "fallback": false,
          "documentation": "When true, disable collisions with all except referenced filteredGroups."
        },
        "physics:mergeGroup": {
          "type": "string",
          "variability": "varying",
          "fallback": "",
          "documentation": "Merge identifier. Groups with identical non-empty mergeGroup are treated as one."
        }
      }
    },
    "PhysicsJoint": {
      "schemaKind": "typed",
      "parent": "Imageable",
      "properties": {
        "physics:body0": {
          "type": "rel",
          "variability": "uniform",
          "documentation": "Relationship to the first Xformable connected by this joint."
        },
        "physics:body1": {
          "type": "rel",
          "variability": "uniform",
          "documentation": "Relationship to the second Xformable connected by this joint."
        },
        "physics:localPos0": {
          "type": "point3f",
          "variability": "varying",
          "fallback": [0, 0, 0],
          "documentation": "Translation of joint frame relative to body0. Units: distance."
        },
        "physics:localPos1": {
          "type": "point3f",
          "variability": "varying",
          "fallback": [0, 0, 0],
          "documentation": "Translation of joint frame relative to body1. Units: distance."
        },
        "physics:localRot0": {
          "type": "quatf",
          "variability": "varying",
          "fallback": [1, 0, 0, 0],
          "documentation": "Orientation of joint frame relative to body0."
        },
        "physics:localRot1": {
          "type": "quatf",
          "variability": "varying",
          "fallback": [1, 0, 0, 0],
          "documentation": "Orientation of joint frame relative to body1."
        },
        "physics:breakForce": {
          "type": "float",
          "variability": "varying",
          "fallback": "inf",
          "documentation": "Maximum force before breaking. Units: mass * distance / second / second."
        },
        "physics:breakTorque": {
          "type": "float",
          "variability": "varying",
          "fallback": "inf",
          "documentation": "Maximum torque before breaking. Units: mass * distance * distance / second / second."
        },
        "physics:jointEnabled": {
          "type": "bool",
          "variability": "varying",
          "fallback": true,
          "documentation": "When false, the joint is not simulated."
        },
        "physics:collisionEnabled": {
          "type": "bool",
          "variability": "varying",
          "fallback": false,
          "documentation": "When true, enables collisions between jointed bodies."
        },
        "physics:excludeFromArticulation": {
          "type": "bool",
          "variability": "uniform",
          "fallback": false,
          "documentation": "When true, excluded from reduced coordinate articulation tree."
        }
      }
    },
    "PhysicsFixedJoint": {
      "schemaKind": "typed",
      "parent": "PhysicsJoint",
      "properties": {}
    },
    "PhysicsDistanceJoint": {
      "schemaKind": "typed",
      "parent": "PhysicsJoint",
      "properties": {
        "physics:minDistance": {
          "type": "float",
          "variability": "varying",
          "fallback": -1.0,
          "documentation": "Minimum permissible distance. Negative means unlimited. Units: distance."
        },
        "physics:maxDistance": {
          "type": "float",
          "variability": "varying",
          "fallback": -1.0,
          "documentation": "Maximum permissible distance. Negative means unlimited. Units: distance."
        }
      }
    },
    "PhysicsSphericalJoint": {
      "schemaKind": "typed",
      "parent": "PhysicsJoint",
      "properties": {
        "physics:axis": {
          "type": "token",
          "variability": "uniform",
          "fallback": "X",
          "documentation": "Principal axis for cone limit definition."
        },
        "physics:coneAngle0Limit": {
          "type": "float",
          "variability": "varying",
          "fallback": -1.0,
          "documentation": "Cone angle toward next axis. Negative means unlimited. Units: degrees."
        },
        "physics:coneAngle1Limit": {
          "type": "float",
          "variability": "varying",
          "fallback": -1.0,
          "documentation": "Cone angle toward third axis. Negative means unlimited. Units: degrees."
        }
      }
    },
    "PhysicsRevoluteJoint": {
      "schemaKind": "typed",
      "parent": "PhysicsJoint",
      "properties": {
        "physics:axis": {
          "type": "token",
          "variability": "uniform",
          "fallback": "X",
          "documentation": "Axis of rotation."
        },
        "physics:lowerLimit": {
          "type": "float",
          "variability": "varying",
          "fallback": "-inf",
          "documentation": "Lower angular limit. Units: degrees."
        },
        "physics:upperLimit": {
          "type": "float",
          "variability": "varying",
          "fallback": "inf",
          "documentation": "Upper angular limit. Units: degrees."
        }
      }
    },
    "PhysicsPrismaticJoint": {
      "schemaKind": "typed",
      "parent": "PhysicsJoint",
      "properties": {
        "physics:axis": {
          "type": "token",
          "variability": "uniform",
          "fallback": "X",
          "documentation": "Axis of translation."
        },
        "physics:lowerLimit": {
          "type": "float",
          "variability": "varying",
          "fallback": "-inf",
          "documentation": "Lower translational limit. Units: distance."
        },
        "physics:upperLimit": {
          "type": "float",
          "variability": "varying",
          "fallback": "inf",
          "documentation": "Upper translational limit. Units: distance."
        }
      }
    },
    "PhysicsRigidBodyAPI": {
      "schemaKind": "singleApply",
      "properties": {
        "physics:velocity": {
          "type": "vector3f",
          "variability": "varying",
          "fallback": [0, 0, 0],
          "documentation": "Linear velocity in local space. Units: distance/second."
        },
        "physics:angularVelocity": {
          "type": "vector3f",
          "variability": "varying",
          "fallback": [0, 0, 0],
          "documentation": "Angular velocity in local space at center of mass. Units: degrees/second."
        },
        "physics:rigidBodyEnabled": {
          "type": "bool",
          "variability": "varying",
          "fallback": true,
          "documentation": "When false, disables all rigid body behavior."
        },
        "physics:kinematicEnabled": {
          "type": "bool",
          "variability": "varying",
          "fallback": false,
          "documentation": "When true, motion is animated rather than simulated."
        },
        "physics:startsAsleep": {
          "type": "bool",
          "variability": "uniform",
          "fallback": false,
          "documentation": "When true, body begins simulation in sleeping state."
        },
        "physics:simulationOwner": {
          "type": "rel",
          "variability": "uniform",
          "documentation": "Relationship to a PhysicsScene prim."
        }
      }
    },
    "PhysicsCollisionAPI": {
      "schemaKind": "singleApply",
      "properties": {
        "physics:collisionEnabled": {
          "type": "bool",
          "variability": "varying",
          "fallback": true,
          "documentation": "When false, disables all collision behavior."
        },
        "physics:simulationOwner": {
          "type": "rel",
          "variability": "uniform",
          "documentation": "Relationship to a PhysicsScene prim."
        }
      }
    },
    "PhysicsMeshCollisionAPI": {
      "schemaKind": "singleApply",
      "properties": {
        "physics:approximation": {
          "type": "token",
          "variability": "uniform",
          "fallback": "none",
          "documentation": "Collision approximation: none, meshSimplification, convexDecomposition, convexHull, boundingSphere, boundingCube."
        }
      }
    },
    "PhysicsMassAPI": {
      "schemaKind": "singleApply",
      "properties": {
        "physics:mass": {
          "type": "float",
          "variability": "varying",
          "fallback": 0.0,
          "documentation": "Total mass override. Zero means ignored. Units: mass."
        },
        "physics:density": {
          "type": "float",
          "variability": "varying",
          "fallback": 0.0,
          "documentation": "Density. Zero means ignored. Units: mass/distance^3."
        },
        "physics:centerOfMass": {
          "type": "point3f",
          "variability": "varying",
          "fallback": [0, 0, 0],
          "documentation": "Center of mass in local space. Units: distance."
        },
        "physics:diagonalInertia": {
          "type": "float3",
          "variability": "varying",
          "fallback": [0, 0, 0],
          "documentation": "Diagonalized inertia tensor. Zero means not specified. Units: mass * distance^2."
        },
        "physics:principalAxes": {
          "type": "quatf",
          "variability": "varying",
          "fallback": [0, 0, 0, 0],
          "documentation": "Orientation of inertia principal axes. Zero quaternion means not specified."
        }
      }
    },
    "PhysicsMaterialAPI": {
      "schemaKind": "singleApply",
      "properties": {
        "physics:staticFriction": {
          "type": "float",
          "variability": "varying",
          "fallback": 0.0,
          "documentation": "Static friction coefficient. Unitless."
        },
        "physics:dynamicFriction": {
          "type": "float",
          "variability": "varying",
          "fallback": 0.0,
          "documentation": "Dynamic friction coefficient. Unitless."
        },
        "physics:restitution": {
          "type": "float",
          "variability": "varying",
          "fallback": 0.0,
          "documentation": "Coefficient of restitution. Unitless."
        },
        "physics:density": {
          "type": "float",
          "variability": "varying",
          "fallback": 0.0,
          "documentation": "Density. Zero means ignored. Units: mass/distance^3."
        }
      }
    },
    "PhysicsFilteredPairsAPI": {
      "schemaKind": "singleApply",
      "properties": {
        "physics:filteredPairs": {
          "type": "rel",
          "variability": "uniform",
          "documentation": "Multi-target relationship to prims with which collisions are disabled."
        }
      }
    },
    "PhysicsArticulationRootAPI": {
      "schemaKind": "singleApply",
      "properties": {}
    },
    "PhysicsLimitAPI": {
      "schemaKind": "multipleApply",
      "instancePrefix": "limit",
      "properties": {
        "limit:<__INSTANCE_NAME__>:physics:low": {
          "type": "float",
          "variability": "varying",
          "fallback": "-inf",
          "documentation": "Lower limit. Units: distance (trans) or degrees (rot)."
        },
        "limit:<__INSTANCE_NAME__>:physics:high": {
          "type": "float",
          "variability": "varying",
          "fallback": "inf",
          "documentation": "Upper limit. Units: distance (trans) or degrees (rot)."
        }
      }
    },
    "PhysicsDriveAPI": {
      "schemaKind": "multipleApply",
      "instancePrefix": "drive",
      "properties": {
        "drive:<__INSTANCE_NAME__>:physics:type": {
          "type": "token",
          "variability": "uniform",
          "fallback": "force",
          "documentation": "Drive type: force or acceleration."
        },
        "drive:<__INSTANCE_NAME__>:physics:stiffness": {
          "type": "float",
          "variability": "varying",
          "fallback": 0.0,
          "documentation": "Spring stiffness coefficient."
        },
        "drive:<__INSTANCE_NAME__>:physics:damping": {
          "type": "float",
          "variability": "varying",
          "fallback": 0.0,
          "documentation": "Damping coefficient."
        },
        "drive:<__INSTANCE_NAME__>:physics:targetPosition": {
          "type": "float",
          "variability": "varying",
          "fallback": 0.0,
          "documentation": "Target position. Units: distance (trans) or degrees (rot)."
        },
        "drive:<__INSTANCE_NAME__>:physics:targetVelocity": {
          "type": "float",
          "variability": "varying",
          "fallback": 0.0,
          "documentation": "Target velocity. Units: distance/s (trans) or degrees/s (rot)."
        },
        "drive:<__INSTANCE_NAME__>:physics:maxForce": {
          "type": "float",
          "variability": "varying",
          "fallback": "inf",
          "documentation": "Maximum force the drive can apply."
        }
      }
    }
  }
})JSON";

    LoadFromJSON(kPhysicsSchemaJSON, &err);
}

} // namespace nanousd
