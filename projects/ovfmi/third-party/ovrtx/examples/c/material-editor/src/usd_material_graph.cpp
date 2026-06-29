// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "usd_material_graph.hpp"

#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/usd/usdShade/nodeGraph.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdr/registry.h>
#include <pxr/usd/sdr/shaderNode.h>
#include <pxr/usd/sdr/shaderProperty.h>
#include <pxr/usd/sdr/sdfTypeIndicator.h>
#include <pxr/usd/sdr/declare.h>
#include <pxr/base/gf/vec2d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/value.h>

#include <cstdint>
#include <iostream>
#include <optional>
#include <unordered_set>

PXR_NAMESPACE_USING_DIRECTIVE

struct ConnectedSource {
    std::string node_path;
    std::string output_name;
};

// ---------------------------------------------------------------------------
// Value extraction helpers
// ---------------------------------------------------------------------------

static ShaderInput::Value extractValue(const VtValue &val, const std::string &type_name)
{
    if (type_name == "float" && val.IsHolding<float>()) {
        return val.UncheckedGet<float>();
    }
    if (type_name == "float" && val.IsHolding<double>()) {
        return static_cast<float>(val.UncheckedGet<double>());
    }
    if (type_name == "int" && val.IsHolding<int>()) {
        return val.UncheckedGet<int>();
    }
    if (type_name == "int" && val.IsHolding<int64_t>()) {
        return static_cast<int>(val.UncheckedGet<int64_t>());
    }
    if (type_name == "bool" && val.IsHolding<bool>()) {
        return val.UncheckedGet<bool>();
    }
    if ((type_name == "color3f" || type_name == "float3") && val.IsHolding<GfVec3f>()) {
        GfVec3f v = val.UncheckedGet<GfVec3f>();
        return std::array<float, 3>{v[0], v[1], v[2]};
    }
    if ((type_name == "color3f" || type_name == "float3") && val.IsHolding<GfVec3d>()) {
        GfVec3d v = val.UncheckedGet<GfVec3d>();
        return std::array<float, 3>{
            static_cast<float>(v[0]),
            static_cast<float>(v[1]),
            static_cast<float>(v[2])};
    }
    if (type_name == "asset" && val.IsHolding<SdfAssetPath>()) {
        SdfAssetPath path = val.UncheckedGet<SdfAssetPath>();
        std::string resolved = path.GetResolvedPath();
        return resolved.empty() ? path.GetAssetPath() : resolved;
    }
    if (type_name == "token" && val.IsHolding<TfToken>()) {
        return val.UncheckedGet<TfToken>().GetString();
    }
    if (type_name == "string" && val.IsHolding<std::string>()) {
        return val.UncheckedGet<std::string>();
    }

    // Fallback: return string representation
    std::ostringstream oss;
    oss << val;
    return oss.str();
}

static std::string simplifyTypeName(const std::string &sdf_type)
{
    if (sdf_type == "float") return "float";
    if (sdf_type == "double") return "float";
    if (sdf_type == "int") return "int";
    if (sdf_type == "integer") return "int";
    if (sdf_type == "bool" || sdf_type == "boolean") return "bool";
    if (sdf_type == "color" || sdf_type == "color3" ||
        sdf_type == "color3f" || sdf_type == "GfVec3f" ||
        sdf_type == "GfVec3d") return "color3f";
    if (sdf_type == "float3" || sdf_type == "vector3" ||
        sdf_type == "normal" || sdf_type == "point3") return "float3";
    if (sdf_type == "float2" || sdf_type == "vector2") return "float2";
    if (sdf_type == "asset" || sdf_type == "filename" ||
        sdf_type == "SdfAssetPath") return "asset";
    if (sdf_type == "token" || sdf_type == "TfToken") return "token";
    if (sdf_type == "string") return "string";
    return sdf_type;
}

// Try to parse a range from an Sdr hint value like "{'max':1,'min':0}"
static bool parseRange(const std::string &hint_val, float &out_min, float &out_max)
{
    // Look for 'min': and 'max': patterns
    bool got_min = false, got_max = false;
    auto extract = [&](const std::string &key, float &out) -> bool {
        size_t pos = hint_val.find(key);
        if (pos == std::string::npos) return false;
        pos += key.size();
        // skip any whitespace or colons
        while (pos < hint_val.size() && (hint_val[pos] == ':' || hint_val[pos] == ' ' || hint_val[pos] == '\''))
            ++pos;
        try {
            out = std::stof(hint_val.substr(pos));
            return true;
        } catch (...) {
            return false;
        }
    };

    got_min = extract("'min'", out_min);
    got_max = extract("'max'", out_max);
    return got_min && got_max;
}

static void extractRange(const SdrShaderPropertyConstPtr &prop, ShaderInput &si)
{
    const SdrTokenMap &hints = prop->GetHints();
    // Prefer soft_range, fall back to hard_range
    for (const TfToken &key : {TfToken("soft_range"), TfToken("hard_range")}) {
        auto it = hints.find(key);
        if (it != hints.end()) {
            float rmin = 0.0f, rmax = 1.0f;
            if (parseRange(it->second, rmin, rmax)) {
                si.has_range = true;
                si.range_min = rmin;
                si.range_max = rmax;
                return;
            }
        }
    }
}

static std::optional<ConnectedSource> getConnectedSource(const UsdShadeInput &input)
{
    UsdShadeConnectableAPI source;
    TfToken source_name;
    UsdShadeAttributeType source_type;
    if (!input.GetConnectedSource(&source, &source_name, &source_type)) {
        return std::nullopt;
    }

    return ConnectedSource{
        source.GetPrim().GetPath().GetString(),
        source_name.GetString()};
}

static void applyConnectedSource(ShaderInput &si, const ConnectedSource &source)
{
    si.is_connected = true;
    si.is_authored = true;
    si.connected_source_node_path = source.node_path;
    si.connected_source_output = source.output_name;
}

static ShaderInput::Value fallbackValueForType(const std::string &type_name)
{
    if (type_name == "float") return 0.0f;
    if (type_name == "int") return 0;
    if (type_name == "bool") return false;
    if (type_name == "color3f" || type_name == "float3") {
        return std::array<float, 3>{0.0f, 0.0f, 0.0f};
    }
    return std::string();
}

// Map Sdr property type tokens to our simplified type names
static std::string sdrTypeToSimplified(const SdrShaderPropertyConstPtr &prop)
{
    // Prefer Sdr type token — it reflects the shader's actual type (e.g. "color3f")
    // even when GetTypeAsSdfType() loses fidelity (e.g. returns "token" for color3f).
    std::string t = simplifyTypeName(prop->GetType().GetString());
    if (t == "color3f" || t == "float3" || t == "float" || t == "int" ||
        t == "bool" || t == "asset" || t == "string" || t == "token" ||
        t == "float2") {
        return t;
    }

    // Fallback to SdfType
    SdrSdfTypeIndicator indicator = prop->GetTypeAsSdfType();
    SdfValueTypeName sdf_type = indicator.GetSdfType();
    if (sdf_type) {
        return simplifyTypeName(sdf_type.GetType().GetTypeName());
    }

    return t;
}

// ---------------------------------------------------------------------------
// Sdr shader node lookup
// ---------------------------------------------------------------------------

static SdrShaderNodeConstPtr findSdrNode(const UsdShadeShader &shader)
{
    static const TfToken source_types[] = {
        TfToken("mdl"),
        TfToken("mtlx"),
        TfToken("materialx"),
        TfToken("glslfx"),
        TfToken("OSL"),
    };

    for (const TfToken &source_type : source_types) {
        SdrShaderNodeConstPtr node = shader.GetShaderNodeForSourceType(source_type);
        if (node) {
            return node;
        }
    }

    // Try all registered source types as fallback
    SdrRegistry &registry = SdrRegistry::GetInstance();
    TfTokenVector all_types = registry.GetAllShaderNodeSourceTypes();
    for (const TfToken &t : all_types) {
        SdrShaderNodeConstPtr node = shader.GetShaderNodeForSourceType(t);
        if (node) {
            return node;
        }
    }

    return nullptr;
}

static void applyNodeGraphUiMetadata(const UsdPrim &prim, ShaderNode &node)
{
    UsdAttribute position_attr =
        prim.GetAttribute(TfToken("ui:nodegraph:node:pos"));
    VtValue position_value;
    if (position_attr && position_attr.Get(&position_value) &&
        !position_value.IsEmpty()) {
        if (position_value.IsHolding<GfVec2f>()) {
            GfVec2f p = position_value.UncheckedGet<GfVec2f>();
            node.has_node_graph_position = true;
            node.node_graph_position = {p[0], p[1]};
        } else if (position_value.IsHolding<GfVec2d>()) {
            GfVec2d p = position_value.UncheckedGet<GfVec2d>();
            node.has_node_graph_position = true;
            node.node_graph_position = {
                static_cast<float>(p[0]),
                static_cast<float>(p[1])};
        }
    }

    UsdAttribute expansion_attr =
        prim.GetAttribute(TfToken("ui:nodegraph:node:expansionState"));
    TfToken expansion_state;
    if (expansion_attr && expansion_attr.Get(&expansion_state)) {
        node.node_graph_expanded = expansion_state == TfToken("open");
    }
}

// ---------------------------------------------------------------------------
// Build shader node with Sdr-driven inputs
// ---------------------------------------------------------------------------

static ShaderNode buildShaderNode(const UsdShadeShader &shader)
{
    ShaderNode node;
    node.prim_path = shader.GetPath().GetString();
    node.display_name = shader.GetPath().GetName();
    applyNodeGraphUiMetadata(shader.GetPrim(), node);

    // Get MDL source asset info
    SdfAssetPath mdl_asset;
    if (shader.GetSourceAsset(&mdl_asset, TfToken("mdl"))) {
        node.mdl_asset = mdl_asset.GetAssetPath();
    }

    TfToken sub_id;
    if (shader.GetSourceAssetSubIdentifier(&sub_id, TfToken("mdl"))) {
        node.mdl_sub_id = sub_id.GetString();
    }
    if (node.mdl_sub_id.empty()) {
        TfToken shader_id;
        if (shader.GetIdAttr().Get(&shader_id)) {
            node.mdl_sub_id = shader_id.GetString();
        }
    }

    // Collect authored input values and names for quick lookup
    std::unordered_map<std::string, VtValue> authored_values;
    std::unordered_map<std::string, SdfValueTypeName> authored_types;
    std::unordered_map<std::string, ConnectedSource> connected_sources;
    for (const UsdShadeInput &input : shader.GetInputs()) {
        std::string input_name = input.GetBaseName().GetString();
        authored_types[input_name] = input.GetTypeName();

        if (std::optional<ConnectedSource> source = getConnectedSource(input)) {
            connected_sources[input_name] = *source;
            continue;
        }

        VtValue val;
        if (input.Get(&val) && !val.IsEmpty()) {
            authored_values[input_name] = val;
        }
    }

    // Try Sdr lookup for full property list
    SdrShaderNodeConstPtr sdr_node = findSdrNode(shader);

    // Dump Sdr info for debugging
    if (sdr_node) {
        std::cerr << "[Sdr] " << node.prim_path
                  << ": " << sdr_node->GetName()
                  << " (" << sdr_node->GetIdentifier() << ")\n";
        for (const TfToken &name : sdr_node->GetShaderInputNames()) {
            SdrShaderPropertyConstPtr prop = sdr_node->GetShaderInput(name);
            if (!prop) continue;
            std::cerr << "  " << name.GetString()
                      << "  type=" << prop->GetType().GetString();
            TfToken label = prop->GetLabel();
            if (!label.IsEmpty())
                std::cerr << "  label=\"" << label.GetString() << "\"";
            TfToken page = prop->GetPage();
            if (!page.IsEmpty())
                std::cerr << "  page=\"" << page.GetString() << "\"";
            const VtValue &def = prop->GetDefaultValue();
            if (!def.IsEmpty())
                std::cerr << "  default=" << def;
            SdrSdfTypeIndicator sdf_ind = prop->GetTypeAsSdfType();
            SdfValueTypeName sdf_type = sdf_ind.GetSdfType();
            if (sdf_type)
                std::cerr << "  sdfType=" << sdf_type.GetType().GetTypeName();
            const SdrTokenMap &hints = prop->GetHints();
            for (const auto &[k, v] : hints)
                std::cerr << "  " << k.GetString() << "=" << v;
            const SdrOptionVec &options = prop->GetOptions();
            if (!options.empty()) {
                std::cerr << "  options=[";
                bool first = true;
                for (const auto &[n, val] : options) {
                    if (!first) std::cerr << ",";
                    std::cerr << n.GetString();
                    first = false;
                }
                std::cerr << "]";
            }
            std::cerr << "\n";
        }
    } else {
        std::cerr << "[Sdr] No shader definition found for " << node.prim_path << "\n";
    }

    if (sdr_node) {
        // Enumerate ALL inputs from Sdr
        const SdrTokenVec &sdr_input_names = sdr_node->GetShaderInputNames();
        std::unordered_set<std::string> seen;

        for (const TfToken &sdr_name : sdr_input_names) {
            SdrShaderPropertyConstPtr prop = sdr_node->GetShaderInput(sdr_name);
            if (!prop) {
                continue;
            }

            std::string name = sdr_name.GetString();
            seen.insert(name);

            ShaderInput si;
            si.name = name;
            si.type_name = sdrTypeToSimplified(prop);
            si.label = prop->GetLabel().GetString();
            si.page = prop->GetPage().GetString();

            const SdrOptionVec &sdr_options = prop->GetOptions();
            for (const auto &[opt_name, opt_val] : sdr_options) {
                si.options.push_back(opt_name.GetString());
            }

            extractRange(prop, si);

            // Check if authored in USD
            auto authored_it = authored_values.find(name);
            if (authored_it != authored_values.end()) {
                si.is_authored = true;
                si.value = extractValue(authored_it->second, si.type_name);
            } else {
                si.is_authored = false;
                // Use Sdr default
                const VtValue &def = prop->GetDefaultValue();
                if (!def.IsEmpty()) {
                    si.value = extractValue(def, si.type_name);
                } else {
                    // No default available — use type-appropriate zero
                    si.value = fallbackValueForType(si.type_name);
                }
            }

            auto connected_it = connected_sources.find(name);
            if (connected_it != connected_sources.end()) {
                applyConnectedSource(si, connected_it->second);
            }

            node.inputs.push_back(std::move(si));
        }

        // Add any authored inputs not in Sdr (shouldn't happen normally, but be safe)
        for (const auto &[name, val] : authored_values) {
            if (seen.count(name)) {
                continue;
            }
            std::string type_name = simplifyTypeName(
                authored_types[name].GetType().GetTypeName());

            ShaderInput si;
            si.name = name;
            si.type_name = type_name;
            si.value = extractValue(val, type_name);
            si.is_authored = true;
            node.inputs.push_back(std::move(si));
        }

        // Add any connected inputs not in Sdr.
        for (const auto &[name, source] : connected_sources) {
            if (seen.count(name)) {
                continue;
            }
            std::string type_name = simplifyTypeName(
                authored_types[name].GetType().GetTypeName());

            ShaderInput si;
            si.name = name;
            si.type_name = type_name;
            si.value = fallbackValueForType(type_name);
            applyConnectedSource(si, source);
            node.inputs.push_back(std::move(si));
        }
    } else {
        // Sdr not available — fall back to authored inputs only
        std::cerr << "[Sdr] No shader definition found for " << node.prim_path << "\n";
        for (const auto &[name, val] : authored_values) {
            std::string type_name = simplifyTypeName(
                authored_types[name].GetType().GetTypeName());

            ShaderInput si;
            si.name = name;
            si.type_name = type_name;
            si.value = extractValue(val, type_name);
            si.is_authored = true;
            node.inputs.push_back(std::move(si));
        }
        for (const auto &[name, source] : connected_sources) {
            std::string type_name = simplifyTypeName(
                authored_types[name].GetType().GetTypeName());

            ShaderInput si;
            si.name = name;
            si.type_name = type_name;
            si.value = fallbackValueForType(type_name);
            applyConnectedSource(si, source);
            node.inputs.push_back(std::move(si));
        }
    }

    // Enumerate outputs
    for (const UsdShadeOutput &output : shader.GetOutputs()) {
        node.output_names.push_back(output.GetBaseName().GetString());
    }

    return node;
}

std::vector<MaterialGraph> loadMaterialGraphs(const std::string &usda_path)
{
    std::vector<MaterialGraph> graphs;

    UsdStageRefPtr stage = UsdStage::Open(usda_path);
    if (!stage) {
        std::cerr << "Failed to open USD stage: " << usda_path << "\n";
        return graphs;
    }

    for (const UsdPrim &prim : stage->Traverse()) {
        if (!prim.IsA<UsdShadeMaterial>()) {
            continue;
        }

        UsdShadeMaterial material(prim);
        MaterialGraph graph;
        graph.material_path = prim.GetPath().GetString();
        graph.display_name = prim.GetPath().GetName();

        // Find shader and node graph children.
        // NodeGraphs are shown as collapsed single nodes — we skip their
        // descendants so the inner shaders don't appear as separate nodes.
        std::unordered_set<std::string> node_graph_paths;
        for (const UsdPrim &child : prim.GetDescendants()) {
            // Skip prims inside a NodeGraph we've already collected
            bool inside_node_graph = false;
            for (const std::string &ng_path : node_graph_paths) {
                if (child.GetPath().GetString().rfind(ng_path + "/", 0) == 0) {
                    inside_node_graph = true;
                    break;
                }
            }
            if (inside_node_graph) {
                continue;
            }

            if (child.IsA<UsdShadeNodeGraph>() &&
                !child.IsA<UsdShadeMaterial>()) {
                // Collect NodeGraph: collapsed node + internal expand data
                std::string ng_path = child.GetPath().GetString();
                node_graph_paths.insert(ng_path);

                NodeGraphData ng_data;

                // Collapsed representation
                ng_data.collapsed_node.prim_path = ng_path;
                ng_data.collapsed_node.display_name = child.GetPath().GetName();
                ng_data.collapsed_node.is_node_graph = true;
                applyNodeGraphUiMetadata(child, ng_data.collapsed_node);

                UsdShadeNodeGraph node_graph(child);

                // Try Sdr lookup for the NodeGraph's inner shader
                // (to get labels and pages for the pass-through inputs)
                SdrShaderNodeConstPtr ng_sdr_node = nullptr;
                for (const UsdPrim &inner : child.GetChildren()) {
                    if (inner.IsA<UsdShadeShader>()) {
                        UsdShadeShader inner_shader(inner);
                        ng_sdr_node = findSdrNode(inner_shader);
                        if (ng_sdr_node) break;
                    }
                }

                for (const UsdShadeInput &input : node_graph.GetInputs()) {
                    ShaderInput si;
                    si.name = input.GetBaseName().GetString();
                    si.type_name = simplifyTypeName(
                        input.GetTypeName().GetType().GetTypeName());

                    // Try Sdr metadata from the inner shader
                    if (ng_sdr_node) {
                        SdrShaderPropertyConstPtr prop =
                            ng_sdr_node->GetShaderInput(TfToken(si.name));
                        if (prop) {
                            si.label = prop->GetLabel().GetString();
                            si.page = prop->GetPage().GetString();
                            if (si.type_name == "unknown" || si.type_name.empty()) {
                                si.type_name = sdrTypeToSimplified(prop);
                            }
                            const SdrOptionVec &sdr_options = prop->GetOptions();
                            for (const auto &[opt_name, opt_val] : sdr_options) {
                                si.options.push_back(opt_name.GetString());
                            }
                            extractRange(prop, si);
                        }
                    }

                    if (std::optional<ConnectedSource> source = getConnectedSource(input)) {
                        applyConnectedSource(si, *source);
                    }

                    VtValue val;
                    if (!si.is_connected && input.Get(&val) && !val.IsEmpty()) {
                        si.value = extractValue(val, si.type_name);
                        si.is_authored = true;
                    } else {
                        // Try Sdr default
                        bool got_default = false;
                        if (ng_sdr_node) {
                            SdrShaderPropertyConstPtr prop =
                                ng_sdr_node->GetShaderInput(TfToken(si.name));
                            if (prop) {
                                const VtValue &def = prop->GetDefaultValue();
                                if (!def.IsEmpty()) {
                                    si.value = extractValue(def, si.type_name);
                                    got_default = true;
                                }
                            }
                        }
                        if (!got_default) {
                            si.value = fallbackValueForType(si.type_name);
                        }
                    }

                    ng_data.collapsed_node.inputs.push_back(std::move(si));
                }
                for (const UsdShadeOutput &output : node_graph.GetOutputs()) {
                    ng_data.collapsed_node.output_names.push_back(
                        output.GetBaseName().GetString());
                }

                // Internal nodes
                for (const UsdPrim &inner : child.GetDescendants()) {
                    if (inner.IsA<UsdShadeShader>()) {
                        UsdShadeShader inner_shader(inner);
                        ng_data.internal_nodes.push_back(buildShaderNode(inner_shader));
                    }
                }

                // Internal connections (between internal nodes and from NG inputs)
                for (const UsdPrim &inner : child.GetDescendants()) {
                    UsdShadeConnectableAPI connectable(inner);
                    if (!connectable) continue;
                    for (const UsdShadeInput &input : connectable.GetInputs()) {
                        UsdShadeConnectableAPI source;
                        TfToken source_name;
                        UsdShadeAttributeType source_type;
                        if (input.GetConnectedSource(&source, &source_name, &source_type)) {
                            GraphConnection conn;
                            conn.source_node_path = source.GetPrim().GetPath().GetString();
                            conn.source_output = source_name.GetString();
                            conn.dest_node_path = inner.GetPath().GetString();
                            conn.dest_input = input.GetBaseName().GetString();
                            ng_data.internal_connections.push_back(std::move(conn));
                        }
                    }
                }

                // Output routing: NG output → internal source
                for (const UsdShadeOutput &output : node_graph.GetOutputs()) {
                    UsdShadeConnectableAPI source;
                    TfToken source_name;
                    UsdShadeAttributeType source_type;
                    if (output.GetConnectedSource(&source, &source_name, &source_type)) {
                        ng_data.output_routing[output.GetBaseName().GetString()] =
                            {source.GetPrim().GetPath().GetString(), source_name.GetString()};
                    }
                }

                // Input routing: NG input → internal destination
                // (found by scanning internal connections where source is the NG prim)
                for (const GraphConnection &ic : ng_data.internal_connections) {
                    if (ic.source_node_path == ng_path) {
                        ng_data.input_routing[ic.source_output] =
                            {ic.dest_node_path, ic.dest_input};
                    }
                }

                // Add collapsed node to the flat list
                graph.shaders.push_back(ng_data.collapsed_node);
                graph.node_graphs[ng_path] = std::move(ng_data);
            } else if (child.IsA<UsdShadeShader>()) {
                UsdShadeShader shader_prim(child);
                graph.shaders.push_back(buildShaderNode(shader_prim));
            }
        }

        // Extract connections: material outputs → shader/nodegraph outputs.
        // Enumerate instead of hardcoding contexts so MDL, MaterialX (mtlx),
        // and plain USD surface outputs are all represented in the graph.
        for (const UsdShadeOutput &mat_output : material.GetOutputs()) {
            UsdShadeConnectableAPI source;
            TfToken source_name;
            UsdShadeAttributeType source_type;
            if (mat_output.GetConnectedSource(&source, &source_name, &source_type)) {
                GraphConnection conn;
                conn.source_node_path = source.GetPrim().GetPath().GetString();
                conn.source_output = source_name.GetString();
                conn.dest_node_path = graph.material_path;
                conn.dest_input = mat_output.GetBaseName().GetString();
                graph.connections.push_back(std::move(conn));
            }
        }

        // Extract connections: shader/nodegraph input → shader/nodegraph output
        for (const UsdPrim &child : prim.GetDescendants()) {
            UsdShadeConnectableAPI connectable(child);
            if (!connectable) {
                continue;
            }
            for (const UsdShadeInput &input : connectable.GetInputs()) {
                UsdShadeConnectableAPI source;
                TfToken source_name;
                UsdShadeAttributeType source_type;
                if (input.GetConnectedSource(&source, &source_name, &source_type)) {
                    GraphConnection conn;
                    conn.source_node_path = source.GetPrim().GetPath().GetString();
                    conn.source_output = source_name.GetString();
                    conn.dest_node_path = child.GetPath().GetString();
                    conn.dest_input = input.GetBaseName().GetString();
                    graph.connections.push_back(std::move(conn));
                }
            }
        }

        graphs.push_back(std::move(graph));
    }

    return graphs;
}

std::string getBoundMaterialPath(const std::string &usda_path,
                                 const std::string &prim_path)
{
    UsdStageRefPtr stage = UsdStage::Open(usda_path);
    if (!stage) {
        std::cerr << "Failed to open USD stage: " << usda_path << "\n";
        return {};
    }

    UsdPrim prim = stage->GetPrimAtPath(SdfPath(prim_path));
    if (!prim) {
        std::cerr << "Failed to find prim: " << prim_path << "\n";
        return {};
    }

    UsdShadeMaterial bound_material =
        UsdShadeMaterialBindingAPI(prim).ComputeBoundMaterial();
    if (bound_material) {
        return bound_material.GetPath().GetString();
    }

    return {};
}
