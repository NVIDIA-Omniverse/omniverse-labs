// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <variant>

/// Represents a single shader input with its name, USD type, and current value.
struct ShaderInput {
    std::string name;       // USD property name, e.g. "diffuse_color_constant"
    std::string type_name;  // SdfValueTypeName string: "color3f", "float", "asset", "token", "bool", "int"

    // Value storage: typed variant for the types we support editing
    using Value = std::variant<
        float,                          // float
        int,                            // int
        bool,                           // bool
        std::array<float, 3>,           // color3f, float3
        std::string                     // asset, token, string
    >;
    Value value;

    // Sdr metadata
    std::string label;      // display name from Sdr (empty = use name)
    std::string page;       // group/page name for accordion grouping
    std::vector<std::string> options; // enum options from Sdr (empty = not an enum)
    bool has_range = false;     // true if soft_range or hard_range is set
    float range_min = 0.0f;
    float range_max = 1.0f;
    bool is_authored = false;   // true if value was authored in USD (vs Sdr default)
    bool is_connected = false;  // true if this input has a UsdShade connection
    std::string connected_source_node_path; // source shader/node path
    std::string connected_source_output;    // source port basename
};

/// Represents a shader node within a material graph.
struct ShaderNode {
    std::string prim_path;      // e.g. "/World/Looks/srf_green_plastic/Shader"
    std::string display_name;   // e.g. "Shader"
    std::string mdl_asset;      // e.g. "OmniPBR.mdl"
    std::string mdl_sub_id;     // e.g. "OmniPBR"
    std::vector<ShaderInput> inputs;
    std::vector<std::string> output_names; // e.g. ["out"]
    bool is_node_graph = false; // true for collapsed NodeGraph prims
    bool node_graph_expanded = false; // true if USD ui:nodegraph says "open"
    bool has_node_graph_position = false;
    std::array<float, 2> node_graph_position{0.0f, 0.0f};
};

/// A connection between two ports in the material graph.
struct GraphConnection {
    std::string source_node_path;   // e.g. "/World/Looks/srf_green_plastic/Shader"
    std::string source_output;      // e.g. "out"
    std::string dest_node_path;     // e.g. "/World/Looks/srf_green_plastic" (the material)
    std::string dest_input;         // e.g. "mdl:surface"
};

/// Stores both the collapsed and expanded representations of a NodeGraph prim.
struct NodeGraphData {
    ShaderNode collapsed_node;               // single-node representation
    std::vector<ShaderNode> internal_nodes;  // inner Shader prims
    std::vector<GraphConnection> internal_connections; // connections between internals

    // Maps NodeGraph output name → (internal prim path, output name).
    // Used to re-route external connections when expanded.
    std::unordered_map<std::string, std::pair<std::string, std::string>> output_routing;

    // Maps NodeGraph input name → (internal prim path, input name).
    std::unordered_map<std::string, std::pair<std::string, std::string>> input_routing;
};

/// Represents a complete material graph: the material prim plus its shader children and connections.
struct MaterialGraph {
    std::string material_path;  // e.g. "/World/Looks/srf_green_plastic"
    std::string display_name;   // e.g. "srf_green_plastic"
    std::vector<ShaderNode> shaders;
    std::vector<GraphConnection> connections;

    /// NodeGraph data keyed by prim path.  Collapsed nodes are in shaders[];
    /// this map stores the expand data needed to toggle.
    std::unordered_map<std::string, NodeGraphData> node_graphs;
};

/// Open a USD stage and extract all material graphs.
/// Returns one MaterialGraph per UsdShadeMaterial found in the stage.
std::vector<MaterialGraph> loadMaterialGraphs(const std::string &usda_path);

/// Return the material bound to a prim in the USD file, or empty if none exists.
std::string getBoundMaterialPath(const std::string &usda_path,
                                 const std::string &prim_path);
