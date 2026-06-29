// fmi_usd_helper — standalone USD parser for ov-fmi
//
// Reads FmuInstance / FmuConnection / FmuMapping prims from a USD stage and
// prints the same JSON structure that parse_fmi_schema.py produces.  Linking
// against the USD .so files bundled in the ovphysx SDK avoids the usd-core pip
// dependency in the main process.
//
// Build: see CMakeLists.txt in this directory.
// Usage: fmi_usd_helper <scene.usda>
//
// NOTE: linking against the ovphysx-bundled USD .so files is not a supported
// public integration pattern.  It works because they are standard OpenUSD
// non-monolithic builds, but NVIDIA does not guarantee this layout across
// releases (see AGENTS.md).

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/value.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/relationship.h>
#include <pxr/usd/usd/stage.h>

#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

static std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n";  break;
            case '\r': out << "\\r";  break;
            case '\t': out << "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out << buf;
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

static std::string json_str(const std::string& s) {
    return "\"" + json_escape(s) + "\"";
}

static std::string json_bool(bool v) {
    return v ? "true" : "false";
}

// Convert a VtValue to a JSON array of numbers.
// Returns "" if the type is not handled (value is omitted from initial_values).
static std::string vt_to_json_array(const VtValue& val) {
    std::ostringstream ss;

    if (val.IsHolding<GfVec3f>()) {
        auto v = val.UncheckedGet<GfVec3f>();
        ss << "[" << v[0] << "," << v[1] << "," << v[2] << "]";
    } else if (val.IsHolding<GfVec3d>()) {
        auto v = val.UncheckedGet<GfVec3d>();
        ss << "[" << v[0] << "," << v[1] << "," << v[2] << "]";
    } else if (val.IsHolding<GfVec2f>()) {
        auto v = val.UncheckedGet<GfVec2f>();
        ss << "[" << v[0] << "," << v[1] << "]";
    } else if (val.IsHolding<GfVec4f>()) {
        auto v = val.UncheckedGet<GfVec4f>();
        ss << "[" << v[0] << "," << v[1] << "," << v[2] << "," << v[3] << "]";
    } else if (val.IsHolding<GfMatrix4d>()) {
        auto m = val.UncheckedGet<GfMatrix4d>();
        ss << "[";
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                ss << (r || c ? "," : "") << m[r][c];
        ss << "]";
    } else if (val.IsHolding<GfMatrix4f>()) {
        auto m = val.UncheckedGet<GfMatrix4f>();
        ss << "[";
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                ss << (r || c ? "," : "") << m[r][c];
        ss << "]";
    } else if (val.IsHolding<float>()) {
        ss << "[" << val.UncheckedGet<float>() << "]";
    } else if (val.IsHolding<double>()) {
        ss << "[" << val.UncheckedGet<double>() << "]";
    } else {
        return "";  // unsupported type — skip
    }
    return ss.str();
}

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

static bool get_bool(const UsdPrim& prim, const char* attr_name, bool def) {
    auto attr = prim.GetAttribute(TfToken(attr_name));
    if (!attr.IsValid()) return def;
    bool v = def;
    attr.Get(&v);
    return v;
}

static std::string get_string(const UsdPrim& prim, const char* attr_name) {
    auto attr = prim.GetAttribute(TfToken(attr_name));
    if (!attr.IsValid()) return "";
    std::string v;
    if (attr.Get(&v)) return v;
    return "";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: fmi_usd_helper <scene.usda|scene.usdc>\n";
        return 1;
    }

    auto stage = UsdStage::Open(argv[1]);
    if (!stage) {
        std::cerr << "fmi_usd_helper: failed to open USD stage: " << argv[1] << "\n";
        return 1;
    }

    // initial_values: prim_path -> {attr_name -> json_array}
    std::map<std::string, std::map<std::string, std::string>> initial_values;

    std::ostringstream instances_json;
    bool first_instance = true;

    for (const auto& prim : stage->GetPrimsWithTypeName(TfToken("FmuInstance"))) {
        // fmi:fmu
        auto fmu_attr = prim.GetAttribute(TfToken("fmi:fmu"));
        if (!fmu_attr.IsValid()) {
            std::cerr << "WARNING: " << prim.GetPath() << " missing fmi:fmu\n";
            continue;
        }
        SdfAssetPath fmu_path;
        if (!fmu_attr.Get(&fmu_path)) {
            std::cerr << "WARNING: " << prim.GetPath() << " fmi:fmu is not an asset path\n";
            continue;
        }
        std::string fmu_resolved = fmu_path.GetResolvedPath();
        if (fmu_resolved.empty()) fmu_resolved = fmu_path.GetAssetPath();

        bool enabled = get_bool(prim, "fmi:enabled", true);

        std::ostringstream conns_json;
        bool first_conn = true;

        for (const auto& child : prim.GetChildren()) {
            if (child.GetTypeName() != TfToken("FmuConnection")) continue;

            bool conn_enabled = get_bool(child, "fmi:enabled", true);

            // fmi:targets relationship
            SdfPathVector targets;
            auto targets_rel = child.GetRelationship(TfToken("fmi:targets"));
            if (targets_rel.IsValid()) targets_rel.GetForwardedTargets(&targets);

            std::ostringstream targets_json;
            bool first_target = true;
            for (const auto& tp : targets) {
                if (!first_target) targets_json << ",";
                targets_json << json_str(tp.GetString());
                first_target = false;
            }

            std::ostringstream mappings_json;
            bool first_mapping = true;

            for (const auto& mc : child.GetChildren()) {
                if (mc.GetTypeName() != TfToken("FmuMapping")) continue;

                std::string fmi_attr  = get_string(mc, "fmi:fmuAttribute");
                std::string usd_attr  = get_string(mc, "fmi:usdAttribute");
                std::string direction = get_string(mc, "fmi:direction");

                if (fmi_attr.empty() || usd_attr.empty() || direction.empty()) {
                    std::cerr << "WARNING: incomplete FmuMapping at "
                              << mc.GetPath() << "\n";
                    continue;
                }

                GfVec2i usd_mapping(0, 0);
                auto um_attr = mc.GetAttribute(TfToken("fmi:usdMapping"));
                if (um_attr.IsValid()) um_attr.Get(&usd_mapping);

                if (!first_mapping) mappings_json << ",";
                first_mapping = false;
                mappings_json << "{"
                    << "\"fmiAttributeName\":" << json_str(fmi_attr) << ","
                    << "\"usdAttributeName\":" << json_str(usd_attr) << ","
                    << "\"direction\":"        << json_str(direction) << ","
                    << "\"usdMapping\":["      << usd_mapping[0] << ","
                                               << usd_mapping[1] << "]"
                    << "}";

                // Capture initial USD attribute values for INPUT mappings
                if (direction == "input") {
                    for (const auto& tp : targets) {
                        auto target = stage->GetPrimAtPath(tp);
                        if (!target.IsValid()) continue;
                        auto attr = target.GetAttribute(TfToken(usd_attr));
                        if (!attr.IsValid()) continue;
                        VtValue val;
                        if (!attr.Get(&val)) continue;
                        std::string arr = vt_to_json_array(val);
                        if (!arr.empty())
                            initial_values[tp.GetString()][usd_attr] = arr;
                    }
                }
            }

            if (!first_conn) conns_json << ",";
            first_conn = false;
            conns_json << "{"
                << "\"enabled\":"  << json_bool(conn_enabled) << ","
                << "\"targets\":["  << targets_json.str() << "],"
                << "\"mappings\":[" << mappings_json.str() << "]"
                << "}";
        }

        if (!first_instance) instances_json << ",";
        first_instance = false;
        std::string pp = prim.GetPath().GetString();
        instances_json << json_str(pp) << ":{"
            << "\"enabled\":" << json_bool(enabled) << ","
            << "\"fmu\":"     << json_str(fmu_resolved) << ","
            << "\"path\":"    << json_str(pp) << ","
            << "\"connections\":[" << conns_json.str() << "]"
            << "}";
    }

    // Build initial_values JSON
    std::ostringstream iv_json;
    bool first_prim = true;
    for (const auto& [prim_path, attrs] : initial_values) {
        if (!first_prim) iv_json << ",";
        first_prim = false;
        iv_json << json_str(prim_path) << ":{";
        bool first_attr = true;
        for (const auto& [attr_name, val_arr] : attrs) {
            if (!first_attr) iv_json << ",";
            first_attr = false;
            iv_json << json_str(attr_name) << ":" << val_arr;
        }
        iv_json << "}";
    }

    std::cout
        << "{\"instances\":{"  << instances_json.str()
        << "},\"initial_values\":{" << iv_json.str()
        << "}}\n";
    return 0;
}
