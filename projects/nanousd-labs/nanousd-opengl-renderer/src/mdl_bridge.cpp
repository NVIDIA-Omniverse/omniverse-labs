// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mdl_bridge.h"

#include <string.h>

#ifndef NUSD_HAVE_MDL_SDK

int nusd_mdl_bridge_available(void)
{
    return 0;
}

int nusd_mdl_bridge_decode(const char* mdl_source_asset,
                           const char* subidentifier,
                           const char* scene_dir,
                           NusdMdlDecoded* out)
{
    (void)mdl_source_asset;
    (void)subidentifier;
    (void)scene_dir;
    if (out) memset(out, 0, sizeof(*out));
    return 0;
}

int nusd_mdl_bridge_decode_with_inputs(const char* mdl_source_asset,
                                       const char* subidentifier,
                                       const char* scene_dir,
                                       const NusdMdlInput* inputs,
                                       int input_count,
                                       NusdMdlDecoded* out)
{
    (void)inputs;
    (void)input_count;
    return nusd_mdl_bridge_decode(mdl_source_asset, subidentifier, scene_dir, out);
}

#else

#include <mi/mdl_sdk.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace fs = std::filesystem;

namespace {

struct BridgeState {
    std::mutex mutex;
    bool load_attempted = false;
    bool started = false;
    bool shutdown_registered = false;
    bool plugins_loaded = false;
#if defined(_WIN32)
    HMODULE dso = nullptr;
#else
    void* dso = nullptr;
#endif
    mi::base::Handle<mi::neuraylib::INeuray> neuray;
    std::vector<std::string> paths_added;
};

static BridgeState g_state;

static void shutdown_bridge()
{
    std::lock_guard<std::mutex> lock(g_state.mutex);
    if (g_state.neuray.is_valid_interface()) {
        if (g_state.started)
            g_state.neuray->shutdown(true);
        g_state.neuray = nullptr;
        g_state.started = false;
    }
#if defined(_WIN32)
    if (g_state.dso) {
        FreeLibrary(g_state.dso);
        g_state.dso = nullptr;
    }
#else
    if (g_state.dso) {
        dlclose(g_state.dso);
        g_state.dso = nullptr;
    }
#endif
}

static std::string strip_asset_token(const char* s)
{
    if (!s) return std::string();
    std::string v(s);
    if (v.size() >= 2 && v.front() == '@' && v.back() == '@')
        v = v.substr(1, v.size() - 2);
    return v;
}

static bool starts_with(const std::string& s, const char* p)
{
    size_t n = strlen(p);
    return s.size() >= n && s.compare(0, n, p) == 0;
}

static std::string normalize_mdl_name(std::string name)
{
    if (starts_with(name, "mdl::"))
        name = name.substr(3);
    if (!starts_with(name, "::") && name.find("::") != std::string::npos)
        name = "::" + name;
    return name;
}

static std::string dirname_of(const std::string& path)
{
    try {
        fs::path p(path);
        if (p.has_parent_path()) return p.parent_path().string();
    } catch (...) {
    }
    return std::string();
}

static void push_unique(std::vector<std::string>& values, const std::string& value)
{
    if (value.empty()) return;
    if (std::find(values.begin(), values.end(), value) == values.end())
        values.push_back(value);
}

static std::vector<std::string> split_env_paths(const char* env)
{
    std::vector<std::string> out;
    if (!env || !env[0]) return out;
    std::string paths(env);
#if defined(_WIN32)
    const char sep = ';';
#else
    const char sep = ':';
#endif
    size_t pos = 0;
    while (pos <= paths.size()) {
        size_t next = paths.find(sep, pos);
        std::string part = paths.substr(pos, next == std::string::npos
                                             ? std::string::npos
                                             : next - pos);
        push_unique(out, part);
        if (next == std::string::npos) break;
        pos = next + 1;
    }
    return out;
}

static std::string loaded_library_dir(const char* requested_lib, void* symbol)
{
    std::string dir = dirname_of(requested_lib ? requested_lib : "");
    if (!dir.empty()) return dir;
#if defined(_WIN32)
    char path[MAX_PATH];
    DWORD n = g_state.dso ? GetModuleFileNameA(g_state.dso, path, MAX_PATH) : 0;
    if (n > 0 && n < MAX_PATH) return dirname_of(path);
#else
    Dl_info info;
    if (symbol && dladdr(symbol, &info) && info.dli_fname)
        return dirname_of(info.dli_fname);
#endif
    return std::string();
}

static void load_sdk_image_plugins(const char* requested_lib, void* symbol)
{
    if (g_state.plugins_loaded || !g_state.neuray.is_valid_interface()) return;
    g_state.plugins_loaded = true;

    mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugins(
        g_state.neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());
    if (!plugins.is_valid_interface()) return;

    std::vector<std::string> dirs = split_env_paths(getenv("NUSD_MDL_PLUGIN_PATH"));
    push_unique(dirs, loaded_library_dir(requested_lib, symbol));
    if (const char* root = getenv("MDL_SDK_ROOT")) {
        try {
            push_unique(dirs, (fs::path(root) / "lib").string());
        } catch (...) {
        }
    }

    const char* plugin_names[] = {
        "nv_openimageio" MI_BASE_DLL_FILE_EXT,
        "dds" MI_BASE_DLL_FILE_EXT,
    };
    for (const std::string& dir : dirs) {
        for (const char* name : plugin_names) {
            std::string path;
            try {
                path = (fs::path(dir) / name).string();
                if (!fs::exists(path)) continue;
            } catch (...) {
                continue;
            }
            mi::Sint32 result = plugins->load_plugin_library(path.c_str());
            if (getenv("NUSD_MAT_DIAG"))
                fprintf(stderr, "[mat_diag] MDL SDK load plugin %s -> %d\n",
                        path.c_str(), (int)result);
        }
    }
}

static std::string resolve_asset_path(const std::string& asset,
                                      const char* scene_dir)
{
    if (asset.empty()) return asset;
    try {
        fs::path p(asset);
        if (p.is_absolute()) return p.lexically_normal().string();
        if (scene_dir && scene_dir[0]) {
            fs::path combined = fs::path(scene_dir) / p;
            return fs::absolute(combined).lexically_normal().string();
        }
    } catch (...) {
    }
    return asset;
}

static void add_unique_path(mi::neuraylib::IMdl_configuration* config,
                            const std::string& path)
{
    if (!config || path.empty()) return;
    std::string norm = path;
    try {
        norm = fs::absolute(path).lexically_normal().string();
    } catch (...) {
    }
    if (std::find(g_state.paths_added.begin(), g_state.paths_added.end(), norm) !=
        g_state.paths_added.end())
        return;
    config->add_mdl_path(norm.c_str());
    config->add_resource_path(norm.c_str());
    g_state.paths_added.push_back(norm);
}

static void add_env_paths(mi::neuraylib::IMdl_configuration* config)
{
    const char* env = getenv("NUSD_MDL_PATH");
    if (!env || !env[0]) return;
    std::string paths(env);
#if defined(_WIN32)
    const char sep = ';';
#else
    const char sep = ':';
#endif
    size_t pos = 0;
    while (pos <= paths.size()) {
        size_t next = paths.find(sep, pos);
        std::string part = paths.substr(pos, next == std::string::npos
                                             ? std::string::npos
                                             : next - pos);
        if (!part.empty()) add_unique_path(config, part);
        if (next == std::string::npos) break;
        pos = next + 1;
    }
}

static bool ensure_started(const char* scene_dir, const std::string& mdl_file)
{
    if (!g_state.load_attempted) {
        g_state.load_attempted = true;
        const char* lib = getenv("NUSD_MDL_SDK_LIBRARY");
        if (!lib || !lib[0]) lib = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;

#if defined(_WIN32)
        g_state.dso = LoadLibraryA(lib);
        void* symbol = g_state.dso ? (void*)GetProcAddress(g_state.dso, "mi_factory") : nullptr;
#else
        g_state.dso = dlopen(lib, RTLD_LAZY);
        void* symbol = g_state.dso ? dlsym(g_state.dso, "mi_factory") : nullptr;
#endif
        if (!symbol) {
            if (getenv("NUSD_MAT_DIAG"))
                fprintf(stderr, "[mat_diag] MDL SDK bridge could not load %s\n", lib);
            return false;
        }
        g_state.neuray = mi::base::Handle<mi::neuraylib::INeuray>(
            mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol));
        if (!g_state.neuray.is_valid_interface()) {
            if (getenv("NUSD_MAT_DIAG"))
                fprintf(stderr, "[mat_diag] MDL SDK bridge could not create INeuray\n");
            return false;
        }
        load_sdk_image_plugins(lib, symbol);
        if (!g_state.shutdown_registered) {
            atexit(shutdown_bridge);
            g_state.shutdown_registered = true;
        }
    }

    if (!g_state.neuray.is_valid_interface()) return false;

    mi::base::Handle<mi::neuraylib::IMdl_configuration> config(
        g_state.neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
    if (config.is_valid_interface()) {
        if (scene_dir && scene_dir[0]) add_unique_path(config.get(), scene_dir);
        add_unique_path(config.get(), dirname_of(mdl_file));
        add_env_paths(config.get());
        if (!g_state.started) {
            config->add_mdl_user_paths();
            config->add_mdl_system_paths();
        }
    }

    if (!g_state.started) {
        mi::Sint32 result = g_state.neuray->start(true);
        if (result != 0) {
            if (getenv("NUSD_MAT_DIAG"))
                fprintf(stderr, "[mat_diag] MDL SDK start failed: %d\n", (int)result);
            return false;
        }
        g_state.started = true;
    }
    return true;
}

static bool split_subidentifier(const std::string& sub,
                                std::string* module_name,
                                std::string* material_name)
{
    if (material_name) material_name->clear();
    if (module_name) module_name->clear();
    if (sub.empty()) return false;
    std::string n = normalize_mdl_name(sub);
    size_t pos = n.rfind("::");
    if (pos == std::string::npos) {
        if (material_name) *material_name = n;
        return true;
    }
    if (material_name) *material_name = n.substr(pos + 2);
    if (module_name) *module_name = n.substr(0, pos);
    return true;
}

static bool derive_module_name(mi::neuraylib::IMdl_impexp_api* impexp,
                               const std::string& source_asset,
                               const std::string& resolved_file,
                               std::string* module_name)
{
    if (!module_name) return false;
    module_name->clear();

    std::string normalized_asset = normalize_mdl_name(source_asset);
    if (starts_with(normalized_asset, "::")) {
        *module_name = normalized_asset;
        return true;
    }

    std::string lower = resolved_file;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return (char)tolower(c); });
    if (lower.size() >= 5 && lower.substr(lower.size() - 5) == ".mdle") {
        *module_name = resolved_file;
        return true;
    }

    if (impexp) {
        mi::base::Handle<const mi::IString> mdl_name(
            impexp->get_mdl_module_name(
                resolved_file.c_str(),
                mi::neuraylib::IMdl_impexp_api::SEARCH_OPTION_USE_SHORTEST));
        if (mdl_name.is_valid_interface() && mdl_name->get_c_str()) {
            *module_name = mdl_name->get_c_str();
            return true;
        }
    }

    try {
        fs::path stem = fs::path(source_asset).stem();
        if (!stem.empty()) {
            *module_name = "::" + stem.string();
            return true;
        }
    } catch (...) {
    }
    return false;
}

static bool constant_float(const mi::neuraylib::ICompiled_material* cm,
                           const char* path,
                           float* out)
{
    if (!cm || !path || !path[0] || !out) return false;
    mi::base::Handle<const mi::neuraylib::IExpression> expr(
        cm->lookup_sub_expression(path));
    if (!expr.is_valid_interface() ||
        expr->get_kind() != mi::neuraylib::IExpression::EK_CONSTANT)
        return false;
    mi::base::Handle<const mi::neuraylib::IExpression_constant> c(
        expr->get_interface<mi::neuraylib::IExpression_constant>());
    if (!c.is_valid_interface()) return false;
    mi::base::Handle<const mi::neuraylib::IValue> value(c->get_value());
    if (!value.is_valid_interface()) return false;

    mi::base::Handle<const mi::neuraylib::IValue_float> vf(
        value->get_interface<mi::neuraylib::IValue_float>());
    if (vf.is_valid_interface()) {
        *out = vf->get_value();
        return true;
    }
    mi::base::Handle<const mi::neuraylib::IValue_double> vd(
        value->get_interface<mi::neuraylib::IValue_double>());
    if (vd.is_valid_interface()) {
        *out = (float)vd->get_value();
        return true;
    }
    return false;
}

static bool constant_color(const mi::neuraylib::ICompiled_material* cm,
                           const char* path,
                           float out[4])
{
    if (!cm || !path || !path[0] || !out) return false;
    mi::base::Handle<const mi::neuraylib::IExpression> expr(
        cm->lookup_sub_expression(path));
    if (!expr.is_valid_interface() ||
        expr->get_kind() != mi::neuraylib::IExpression::EK_CONSTANT)
        return false;
    mi::base::Handle<const mi::neuraylib::IExpression_constant> c(
        expr->get_interface<mi::neuraylib::IExpression_constant>());
    if (!c.is_valid_interface()) return false;
    mi::base::Handle<const mi::neuraylib::IValue> value(c->get_value());
    if (!value.is_valid_interface()) return false;

    mi::base::Handle<const mi::neuraylib::IValue_color> color(
        value->get_interface<mi::neuraylib::IValue_color>());
    if (color.is_valid_interface()) {
        for (mi::Size i = 0; i < 3; ++i) {
            mi::base::Handle<const mi::neuraylib::IValue_float> component(
                color->get_value(i));
            if (!component.is_valid_interface()) return false;
            out[i] = component->get_value();
        }
        out[3] = 1.0f;
        return true;
    }

    mi::base::Handle<const mi::neuraylib::IValue_vector> vector(
        value->get_interface<mi::neuraylib::IValue_vector>());
    if (vector.is_valid_interface() && vector->get_size() >= 3) {
        for (mi::Size i = 0; i < 3; ++i) {
            mi::base::Handle<const mi::neuraylib::IValue_float> component(
                vector->get_value<mi::neuraylib::IValue_float>(i));
            if (!component.is_valid_interface()) return false;
            out[i] = component->get_value();
        }
        out[3] = 1.0f;
        return true;
    }
    return false;
}

static const mi::neuraylib::IExpression_direct_call* to_direct_call(
    const mi::neuraylib::IExpression* expr,
    const mi::neuraylib::ICompiled_material* cm)
{
    if (!expr) return nullptr;
    if (expr->get_kind() == mi::neuraylib::IExpression::EK_DIRECT_CALL)
        return expr->get_interface<mi::neuraylib::IExpression_direct_call>();
    if (expr->get_kind() == mi::neuraylib::IExpression::EK_TEMPORARY) {
        mi::base::Handle<const mi::neuraylib::IExpression_temporary> temp(
            expr->get_interface<mi::neuraylib::IExpression_temporary>());
        if (!temp.is_valid_interface()) return nullptr;
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> ref(
            cm->get_temporary<mi::neuraylib::IExpression_direct_call>(
                temp->get_index()));
        if (!ref.is_valid_interface()) return nullptr;
        ref->retain();
        return ref.get();
    }
    return nullptr;
}

static const mi::neuraylib::IExpression_direct_call* argument_as_call(
    const mi::neuraylib::ICompiled_material* cm,
    const mi::neuraylib::IExpression_direct_call* call,
    const char* argument_name)
{
    if (!cm || !call || !argument_name) return nullptr;
    mi::base::Handle<const mi::neuraylib::IExpression_list> args(
        call->get_arguments());
    if (!args.is_valid_interface()) return nullptr;
    mi::base::Handle<const mi::neuraylib::IExpression> arg(
        args->get_expression(argument_name));
    return to_direct_call(arg.get(), cm);
}

static const mi::neuraylib::IExpression_direct_call* lookup_call(
    const char* path,
    const mi::neuraylib::ICompiled_material* cm,
    const mi::neuraylib::IExpression_direct_call* parent_call = nullptr)
{
    if (!path || !cm) return nullptr;
    mi::base::Handle<const mi::neuraylib::IExpression_direct_call> result;
    if (!parent_call) {
        mi::base::Handle<const mi::neuraylib::IExpression> expr(
            cm->lookup_sub_expression(path));
        result = mi::base::Handle<const mi::neuraylib::IExpression_direct_call>(
            to_direct_call(expr.get(), cm));
    } else {
        result = mi::base::make_handle_dup(parent_call);
        std::string remaining(path);
        while (!remaining.empty()) {
            size_t dot = remaining.find('.');
            std::string token = dot == std::string::npos
                                    ? remaining
                                    : remaining.substr(0, dot);
            result = mi::base::Handle<const mi::neuraylib::IExpression_direct_call>(
                argument_as_call(cm, result.get(), token.c_str()));
            if (!result.is_valid_interface()) return nullptr;
            if (dot == std::string::npos) break;
            remaining = remaining.substr(dot + 1);
        }
    }
    if (!result.is_valid_interface()) return nullptr;
    result->retain();
    return result.get();
}

static mi::neuraylib::IFunction_definition::Semantics call_semantic(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::IExpression_direct_call* call)
{
    if (!transaction || !call)
        return mi::neuraylib::IFunction_definition::DS_UNKNOWN;
    mi::base::Handle<const mi::neuraylib::IFunction_definition> def(
        transaction->access<mi::neuraylib::IFunction_definition>(
            call->get_definition()));
    if (!def.is_valid_interface())
        return mi::neuraylib::IFunction_definition::DS_UNKNOWN;
    return def->get_semantic();
}

struct PbrPaths {
    std::string base_color;
    std::string metallic;
    std::string roughness;
    std::string opacity;
    std::string clearcoat;
    std::string clearcoat_roughness;
    std::string specular;
    std::string specular_normal_reflectivity;
    std::string transmission_weight;
    std::string transmission_color;
    std::string ior;
};

static void discover_distilled_paths(mi::neuraylib::ITransaction* transaction,
                                     const mi::neuraylib::ICompiled_material* cm,
                                     bool transmissive,
                                     PbrPaths* paths)
{
    if (!paths) return;
    paths->opacity = "geometry.cutout_opacity";
    paths->ior = "ior";

    mi::base::Handle<const mi::neuraylib::IExpression_direct_call> parent(
        lookup_call("surface.scattering", cm));
    mi::neuraylib::IFunction_definition::Semantics semantic =
        call_semantic(transaction, parent.get());
    std::string prefix = "surface.scattering.";

    if (semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER) {
        paths->clearcoat = prefix + "weight";
        paths->clearcoat_roughness = prefix + "layer.roughness_u";
        parent = mi::base::Handle<const mi::neuraylib::IExpression_direct_call>(
            lookup_call("base", cm, parent.get()));
        semantic = call_semantic(transaction, parent.get());
        prefix += "base.";
    }

    if (semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_WEIGHTED_LAYER) {
        parent = mi::base::Handle<const mi::neuraylib::IExpression_direct_call>(
            lookup_call("layer", cm, parent.get()));
        semantic = call_semantic(transaction, parent.get());
        prefix += "layer.";
    }

    if (semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_NORMALIZED_MIX) {
        paths->metallic = prefix + "components.value1.weight";
        paths->base_color = prefix + "components.value1.component.tint";
        paths->roughness = transmissive
                               ? prefix + "components.value1.component.roughness_u.s.r.roughness"
                               : prefix + "components.value1.component.roughness_u";
        parent = mi::base::Handle<const mi::neuraylib::IExpression_direct_call>(
            lookup_call("components.value0.component", cm, parent.get()));
        semantic = call_semantic(transaction, parent.get());
        prefix += "components.value0.component.";
    }

    if (semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER) {
        paths->specular = prefix + "weight";
        paths->specular_normal_reflectivity = prefix + "normal_reflectivity";
        paths->roughness = transmissive
                               ? prefix + "layer.roughness_u.s.r.roughness"
                               : prefix + "layer.roughness_u";
        parent = mi::base::Handle<const mi::neuraylib::IExpression_direct_call>(
            lookup_call("base", cm, parent.get()));
        semantic = call_semantic(transaction, parent.get());
        prefix += "base.";
    }

    if (transmissive &&
        semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_NORMALIZED_MIX) {
        paths->transmission_weight = prefix + "components.value1.weight";
        paths->transmission_color = prefix + "components.value1.component.tint";
        parent = mi::base::Handle<const mi::neuraylib::IExpression_direct_call>(
            lookup_call("components.value0.component", cm, parent.get()));
        semantic = call_semantic(transaction, parent.get());
        prefix += "components.value0.component.";
    }

    if (semantic ==
        mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF) {
        if (paths->metallic.empty()) paths->metallic = "__constant_one__";
        if (paths->roughness.empty()) paths->roughness = prefix + "roughness_u";
        if (paths->base_color.empty()) paths->base_color = prefix + "tint";
    } else if (semantic ==
               mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF) {
        if (paths->base_color.empty()) paths->base_color = prefix + "tint";
        if (paths->roughness.empty()) paths->roughness = prefix + "roughness";
    }
}

static bool apply_paths(const mi::neuraylib::ICompiled_material* cm,
                        const PbrPaths& paths,
                        NusdMdlDecoded* out)
{
    bool wrote = false;
    if (!paths.base_color.empty() &&
        constant_color(cm, paths.base_color.c_str(), out->base_color)) {
        out->has_base_color = 1;
        wrote = true;
    }
    if (!paths.roughness.empty() &&
        constant_float(cm, paths.roughness.c_str(), &out->roughness)) {
        out->has_roughness = 1;
        wrote = true;
    }
    if (!paths.metallic.empty()) {
        if (paths.metallic == "__constant_one__") {
            out->metallic = 1.0f;
            out->has_metallic = 1;
            wrote = true;
        } else if (constant_float(cm, paths.metallic.c_str(), &out->metallic)) {
            out->has_metallic = 1;
            wrote = true;
        }
    }
    if (!paths.opacity.empty() &&
        constant_float(cm, paths.opacity.c_str(), &out->opacity)) {
        out->has_opacity = 1;
        wrote = true;
    }
    if (!paths.ior.empty() && constant_float(cm, paths.ior.c_str(), &out->ior)) {
        out->has_ior = 1;
        out->transmission_ior = out->ior;
        out->has_transmission_ior = 1;
        wrote = true;
    }
    if (!paths.clearcoat.empty() &&
        constant_float(cm, paths.clearcoat.c_str(), &out->clearcoat)) {
        out->has_clearcoat = 1;
        wrote = true;
    }
    if (!paths.clearcoat_roughness.empty() &&
        constant_float(cm, paths.clearcoat_roughness.c_str(),
                       &out->clearcoat_roughness)) {
        out->has_clearcoat_roughness = 1;
        wrote = true;
    }
    if (!paths.specular.empty() &&
        constant_float(cm, paths.specular.c_str(), &out->specular_color[0])) {
        float specular_weight = out->specular_color[0];
        float f0 = specular_weight;
        float normal_reflectivity = 0.0f;
        if (!paths.specular_normal_reflectivity.empty() &&
            constant_float(cm, paths.specular_normal_reflectivity.c_str(),
                           &normal_reflectivity)) {
            f0 = specular_weight * normal_reflectivity;
            if (getenv("NUSD_MAT_DIAG")) {
                fprintf(stderr,
                        "[mat_diag] MDL custom_curve_layer F0: "
                        "weight=%.4f normal_reflectivity=%.4f f0=%.4f\n",
                        specular_weight, normal_reflectivity, f0);
            }
        } else {
            /* Omni/UE-style MDL materials expose `specular` as a 0..1
             * dielectric reflectance control where the default 0.5 maps to
             * F0 ~= 0.04, not literal F0=0.5. MDL distillation often leaves
             * this as a custom_curve_layer weight without an explicit
             * normal_reflectivity input. */
            f0 = specular_weight * 0.08f;
            if (getenv("NUSD_MAT_DIAG")) {
                fprintf(stderr,
                        "[mat_diag] MDL custom_curve_layer F0: "
                        "weight=%.4f default_normal_reflectivity=0.0800 "
                        "f0=%.4f\n",
                        specular_weight, f0);
            }
        }
        f0 = std::clamp(f0, 0.0f, 1.0f);
        out->specular_color[0] = f0;
        out->specular_color[1] = f0;
        out->specular_color[2] = f0;
        out->specular_color[3] = 1.0f;
        out->use_specular_workflow = 1;
        out->has_specular_color = 1;
        out->has_specular_workflow = 1;
        wrote = true;
    }
    if (!paths.transmission_weight.empty() &&
        constant_float(cm, paths.transmission_weight.c_str(),
                       &out->transmission_weight)) {
        out->has_transmission_weight = 1;
        wrote = true;
    }
    if (!paths.transmission_color.empty() &&
        constant_color(cm, paths.transmission_color.c_str(),
                       out->transmission_color)) {
        out->has_transmission_color = 1;
        wrote = true;
    }
    return wrote;
}

static std::string texture_file_from_db_name(
    mi::neuraylib::ITransaction* transaction,
    const char* texture_db_name)
{
    if (!transaction || !texture_db_name || !texture_db_name[0])
        return std::string();

    mi::base::Handle<const mi::neuraylib::ITexture> texture(
        transaction->access<mi::neuraylib::ITexture>(texture_db_name));
    if (!texture.is_valid_interface()) return std::string();

    mi::base::Handle<const mi::neuraylib::IImage> image(
        transaction->access<mi::neuraylib::IImage>(texture->get_image()));
    if (!image.is_valid_interface()) return std::string();

    const char* filename = image->get_filename(0, 0);
    if (filename && filename[0]) return filename;
    const char* original = image->get_original_filename();
    if (original && original[0]) return original;
    return std::string();
}

static void copy_decoded_string(char* dst, size_t dst_size,
                                const std::string& value)
{
    if (!dst || dst_size == 0) return;
    snprintf(dst, dst_size, "%s", value.c_str());
}

static bool add_decoded_texture(NusdMdlDecoded* out,
                                const std::string& name_hint,
                                const std::string& file_path,
                                const std::string& db_name)
{
    if (!out) return false;
    if (file_path.empty() && db_name.empty()) return false;

    for (int i = 0; i < out->texture_count; ++i) {
        const NusdMdlDecodedTexture& existing = out->textures[i];
        if (!file_path.empty() && strcmp(existing.file_path, file_path.c_str()) == 0)
            return false;
        if (file_path.empty() && !db_name.empty() &&
            strcmp(existing.db_name, db_name.c_str()) == 0)
            return false;
    }
    if (out->texture_count >= NUSD_MDL_MAX_DECODED_TEXTURES)
        return false;

    NusdMdlDecodedTexture& tex = out->textures[out->texture_count++];
    copy_decoded_string(tex.name_hint, sizeof(tex.name_hint), name_hint);
    copy_decoded_string(tex.file_path, sizeof(tex.file_path), file_path);
    copy_decoded_string(tex.db_name, sizeof(tex.db_name), db_name);
    return true;
}

static void collect_texture_value(mi::neuraylib::ITransaction* transaction,
                                  const mi::neuraylib::IValue* value,
                                  const std::string& name_hint,
                                  NusdMdlDecoded* out,
                                  int depth)
{
    if (!transaction || !value || !out || depth > 24) return;

    if (value->get_kind() == mi::neuraylib::IValue::VK_TEXTURE) {
        mi::base::Handle<const mi::neuraylib::IValue_texture> texture_value(
            value->get_interface<mi::neuraylib::IValue_texture>());
        if (!texture_value.is_valid_interface()) return;
        const char* db_name_c = texture_value->get_value();
        const char* file_path_c = texture_value->get_file_path();
        std::string db_name = db_name_c ? db_name_c : "";
        std::string file_path = file_path_c ? file_path_c : "";
        if (file_path.empty())
            file_path = texture_file_from_db_name(transaction, db_name.c_str());
        if (add_decoded_texture(out, name_hint, file_path, db_name) &&
            getenv("NUSD_MAT_DIAG")) {
            fprintf(stderr,
                    "[mat_diag] MDL SDK resource texture hint=%s file=%s db=%s\n",
                    name_hint.c_str(), file_path.c_str(), db_name.c_str());
        }
        return;
    }

    mi::base::Handle<const mi::neuraylib::IValue_compound> compound(
        value->get_interface<mi::neuraylib::IValue_compound>());
    if (!compound.is_valid_interface()) return;
    for (mi::Size i = 0, n = compound->get_size(); i < n; ++i) {
        mi::base::Handle<const mi::neuraylib::IValue> child(
            compound->get_value(i));
        collect_texture_value(transaction, child.get(), name_hint, out, depth + 1);
    }
}

static void collect_textures_from_expression(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::ICompiled_material* cm,
    const mi::neuraylib::IExpression* expr,
    const std::string& name_hint,
    NusdMdlDecoded* out,
    int depth)
{
    if (!transaction || !cm || !expr || !out || depth > 32) return;

    switch (expr->get_kind()) {
    case mi::neuraylib::IExpression::EK_CONSTANT: {
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c(
            expr->get_interface<mi::neuraylib::IExpression_constant>());
        if (!c.is_valid_interface()) return;
        mi::base::Handle<const mi::neuraylib::IValue> value(c->get_value());
        collect_texture_value(transaction, value.get(), name_hint, out, depth + 1);
        break;
    }
    case mi::neuraylib::IExpression::EK_TEMPORARY: {
        mi::base::Handle<const mi::neuraylib::IExpression_temporary> temp(
            expr->get_interface<mi::neuraylib::IExpression_temporary>());
        if (!temp.is_valid_interface()) return;
        mi::base::Handle<const mi::neuraylib::IExpression> target(
            cm->get_temporary(temp->get_index()));
        collect_textures_from_expression(transaction, cm, target.get(),
                                         name_hint, out, depth + 1);
        break;
    }
    case mi::neuraylib::IExpression::EK_DIRECT_CALL: {
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> call(
            expr->get_interface<mi::neuraylib::IExpression_direct_call>());
        if (!call.is_valid_interface()) return;
        mi::base::Handle<const mi::neuraylib::IExpression_list> args(
            call->get_arguments());
        if (!args.is_valid_interface()) return;
        for (mi::Size i = 0, n = args->get_size(); i < n; ++i) {
            const char* arg_name = args->get_name(i);
            std::string child_hint = name_hint;
            if (arg_name && arg_name[0]) {
                if (!child_hint.empty()) child_hint += ".";
                child_hint += arg_name;
            }
            mi::base::Handle<const mi::neuraylib::IExpression> arg(
                args->get_expression(i));
            collect_textures_from_expression(transaction, cm, arg.get(),
                                             child_hint, out, depth + 1);
        }
        break;
    }
    default:
        break;
    }
}

static void collect_textures_from_path(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::ICompiled_material* cm,
    const std::string& path,
    NusdMdlDecoded* out)
{
    if (path.empty()) return;
    mi::base::Handle<const mi::neuraylib::IExpression> expr(
        cm->lookup_sub_expression(path.c_str()));
    collect_textures_from_expression(transaction, cm, expr.get(), path, out, 0);
}

static void collect_sdk_textures(mi::neuraylib::ITransaction* transaction,
                                 const mi::neuraylib::ICompiled_material* cm,
                                 const PbrPaths& paths,
                                 NusdMdlDecoded* out)
{
    if (!transaction || !cm || !out) return;

    collect_textures_from_path(transaction, cm, paths.base_color, out);
    collect_textures_from_path(transaction, cm, paths.roughness, out);
    collect_textures_from_path(transaction, cm, paths.metallic, out);
    collect_textures_from_path(transaction, cm, paths.opacity, out);
    collect_textures_from_path(transaction, cm, paths.clearcoat, out);
    collect_textures_from_path(transaction, cm, paths.clearcoat_roughness, out);
    collect_textures_from_path(transaction, cm, paths.specular, out);
    collect_textures_from_path(transaction, cm, paths.transmission_weight, out);
    collect_textures_from_path(transaction, cm, paths.transmission_color, out);

    /* Follows NVIDIA's distilling_glsl/execution_glsl resource path:
     * resolve SDK texture DB names to ITexture -> IImage resources rather than
     * scanning MDL source text. These broad roots catch resources hidden behind
     * target-model expressions until full baking/target-code execution lands. */
    collect_textures_from_path(transaction, cm, "surface.scattering", out);
    collect_textures_from_path(transaction, cm, "surface.emission.emission", out);
    collect_textures_from_path(transaction, cm, "geometry.cutout_opacity", out);
    collect_textures_from_path(transaction, cm, "geometry.normal", out);
}

static std::string normalize_param_name(const char* name)
{
    std::string out;
    if (!name) return out;
    for (const char* p = name; *p; ++p) {
        unsigned char c = (unsigned char)*p;
        if (c == '_' || c == '-' || c == ':') continue;
        out.push_back((char)tolower(c));
    }
    return out;
}

static const NusdMdlInput* find_input_override(const NusdMdlInput* inputs,
                                               int input_count,
                                               const char* parameter_name)
{
    if (!inputs || input_count <= 0 || !parameter_name || !parameter_name[0])
        return nullptr;
    for (int i = 0; i < input_count; ++i) {
        if (inputs[i].name && strcmp(inputs[i].name, parameter_name) == 0)
            return &inputs[i];
    }
    std::string wanted = normalize_param_name(parameter_name);
    for (int i = 0; i < input_count; ++i) {
        if (inputs[i].name && normalize_param_name(inputs[i].name) == wanted)
            return &inputs[i];
    }
    return nullptr;
}

static mi::neuraylib::IValue* create_override_value(
    mi::neuraylib::IValue_factory* values,
    const mi::neuraylib::IType* type,
    const NusdMdlInput& input)
{
    if (!values || !type) return nullptr;
    switch (type->get_kind()) {
    case mi::neuraylib::IType::TK_FLOAT:
        return values->create_float(input.values[0]);
    case mi::neuraylib::IType::TK_DOUBLE:
        return values->create_double((mi::Float64)input.values[0]);
    case mi::neuraylib::IType::TK_INT:
        return values->create_int(input.kind == NUSD_MDL_INPUT_INT
                                      ? input.int_value
                                      : (mi::Sint32)input.values[0]);
    case mi::neuraylib::IType::TK_BOOL:
        return values->create_bool(input.kind == NUSD_MDL_INPUT_BOOL
                                       ? input.int_value != 0
                                       : input.values[0] != 0.0f);
    case mi::neuraylib::IType::TK_COLOR:
        return values->create_color(input.values[0], input.values[1],
                                    input.values[2]);
    case mi::neuraylib::IType::TK_VECTOR: {
        mi::base::Handle<const mi::neuraylib::IType_vector> vector_type(
            type->get_interface<mi::neuraylib::IType_vector>());
        if (!vector_type.is_valid_interface()) return nullptr;
        mi::base::Handle<mi::neuraylib::IValue_vector> vector_value(
            values->create_vector(vector_type.get()));
        if (!vector_value.is_valid_interface()) return nullptr;
        mi::base::Handle<const mi::neuraylib::IType_atomic> element_type(
            vector_type->get_element_type());
        mi::neuraylib::IType::Kind element_kind =
            element_type.is_valid_interface()
                ? element_type->get_kind()
                : mi::neuraylib::IType::TK_FLOAT;
        mi::Size n = vector_value->get_size();
        if (n > 4) n = 4;
        for (mi::Size i = 0; i < n; ++i) {
            if (element_kind == mi::neuraylib::IType::TK_BOOL) {
                mi::base::Handle<mi::neuraylib::IValue_bool> component(
                    vector_value->get_value<mi::neuraylib::IValue_bool>(i));
                if (component.is_valid_interface())
                    component->set_value(input.values[i] != 0.0f);
            } else if (element_kind == mi::neuraylib::IType::TK_INT) {
                mi::base::Handle<mi::neuraylib::IValue_int> component(
                    vector_value->get_value<mi::neuraylib::IValue_int>(i));
                if (component.is_valid_interface())
                    component->set_value((mi::Sint32)input.values[i]);
            } else {
                mi::base::Handle<mi::neuraylib::IValue_float> component(
                    vector_value->get_value<mi::neuraylib::IValue_float>(i));
                if (component.is_valid_interface())
                    component->set_value(input.values[i]);
            }
        }
        vector_value->retain();
        return vector_value.get();
    }
    default:
        return nullptr;
    }
}

static mi::neuraylib::IExpression_list* create_override_arguments(
    mi::neuraylib::IMdl_factory* factory,
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::IFunction_definition* definition,
    const NusdMdlInput* inputs,
    int input_count)
{
    if (!factory || !transaction || !definition || !inputs || input_count <= 0)
        return nullptr;
    mi::base::Handle<mi::neuraylib::IValue_factory> values(
        factory->create_value_factory(transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> expressions(
        factory->create_expression_factory(transaction));
    mi::base::Handle<const mi::neuraylib::IType_list> parameter_types(
        definition->get_parameter_types());
    if (!values.is_valid_interface() || !expressions.is_valid_interface() ||
        !parameter_types.is_valid_interface())
        return nullptr;

    mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
        expressions->create_expression_list());
    bool any = false;
    for (mi::Size i = 0, n = definition->get_parameter_count(); i < n; ++i) {
        const char* parameter_name = definition->get_parameter_name(i);
        const NusdMdlInput* input =
            find_input_override(inputs, input_count, parameter_name);
        if (!input) continue;

        mi::base::Handle<const mi::neuraylib::IType> type(
            parameter_types->get_type(parameter_name));
        mi::base::Handle<mi::neuraylib::IValue> value(
            create_override_value(values.get(), type.get(), *input));
        if (!value.is_valid_interface()) continue;

        mi::base::Handle<const mi::neuraylib::IExpression_constant> expr(
            expressions->create_constant(value.get()));
        if (!expr.is_valid_interface()) continue;
        if (arguments->add_expression(parameter_name, expr.get()) == 0)
            any = true;
    }
    if (!any) return nullptr;
    arguments->retain();
    return arguments.get();
}

static bool decode_with_sdk(const char* mdl_source_asset,
                            const char* subidentifier,
                            const char* scene_dir,
                            const NusdMdlInput* inputs,
                            int input_count,
                            NusdMdlDecoded* out)
{
    std::string source_asset = strip_asset_token(mdl_source_asset);
    if (source_asset.empty()) return false;
    std::string resolved = resolve_asset_path(source_asset, scene_dir);

    if (!ensure_started(scene_dir, resolved)) return false;

    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> impexp(
        g_state.neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> factory(
        g_state.neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IDatabase> database(
        g_state.neuray->get_api_component<mi::neuraylib::IDatabase>());
    if (!impexp.is_valid_interface() || !factory.is_valid_interface() ||
        !database.is_valid_interface())
        return false;

    mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
    mi::base::Handle<mi::neuraylib::ITransaction> transaction(
        scope->create_transaction());
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        factory->create_execution_context());

    std::string module_from_sub;
    std::string material_simple;
    split_subidentifier(subidentifier ? subidentifier : "",
                        &module_from_sub, &material_simple);

    std::string module_name = module_from_sub;
    if (module_name.empty() &&
        !derive_module_name(impexp.get(), source_asset, resolved, &module_name))
        return false;
    if (material_simple.empty()) material_simple = "main";

    mi::Sint32 load_result = impexp->load_module(
        transaction.get(), module_name.c_str(), context.get());
    if (load_result < 0) {
        if (getenv("NUSD_MAT_DIAG"))
            fprintf(stderr, "[mat_diag] MDL load_module failed module=%s result=%d\n",
                    module_name.c_str(), (int)load_result);
        transaction->abort();
        return false;
    }

    mi::base::Handle<const mi::IString> module_db_name(
        factory->get_db_module_name(module_name.c_str()));
    if (!module_db_name.is_valid_interface()) {
        transaction->abort();
        return false;
    }

    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));
    if (!module.is_valid_interface()) {
        transaction->abort();
        return false;
    }

    std::string definition_db_name;
    if (material_simple == "main" && module->get_material_count() > 0) {
        const char* first = module->get_material(0);
        if (first) definition_db_name = first;
    }
    if (definition_db_name.empty()) {
        definition_db_name = std::string(module_db_name->get_c_str()) +
                             "::" + material_simple;
        if (definition_db_name.back() != ')') {
            mi::base::Handle<const mi::IArray> overloads(
                module->get_function_overloads(definition_db_name.c_str()));
            if (overloads.is_valid_interface() && overloads->get_length() == 1) {
                mi::base::Handle<const mi::IString> overload(
                    overloads->get_element<mi::IString>(0));
                if (overload.is_valid_interface())
                    definition_db_name = overload->get_c_str();
            }
        }
    }

    mi::base::Handle<const mi::neuraylib::IFunction_definition> definition(
        transaction->access<mi::neuraylib::IFunction_definition>(
            definition_db_name.c_str()));
    if (!definition.is_valid_interface()) {
        if (getenv("NUSD_MAT_DIAG"))
            fprintf(stderr, "[mat_diag] MDL material definition not found: %s\n",
                    definition_db_name.c_str());
        transaction->abort();
        return false;
    }

    mi::base::Handle<mi::neuraylib::IExpression_list> override_arguments(
        create_override_arguments(factory.get(), transaction.get(),
                                  definition.get(), inputs, input_count));
    if (getenv("NUSD_MAT_DIAG") && override_arguments.is_valid_interface()) {
        fprintf(stderr, "[mat_diag] MDL SDK applying %u authored input override(s)\n",
                (unsigned)override_arguments->get_size());
    }

    mi::Sint32 call_result = 0;
    mi::base::Handle<mi::neuraylib::IFunction_call> call(
        definition->create_function_call(override_arguments.get(), &call_result));
    if (call_result != 0 || !call.is_valid_interface()) {
        transaction->abort();
        return false;
    }

    mi::base::Handle<mi::neuraylib::IType_factory> type_factory(
        factory->create_type_factory(transaction.get()));
    mi::base::Handle<const mi::neuraylib::IType> material_type(
        type_factory->get_predefined_struct(
            mi::neuraylib::IType_struct::SID_MATERIAL));
    context->set_option("target_type", material_type.get());

    mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance(
        call->get_interface<mi::neuraylib::IMaterial_instance>());
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled(
        material_instance->create_compiled_material(
            mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS,
            context.get()));
    if (!compiled.is_valid_interface()) {
        transaction->abort();
        return false;
    }

    bool wrote = false;
    mi::base::Handle<mi::neuraylib::IMdl_distiller_api> distiller(
        g_state.neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());
    if (distiller.is_valid_interface()) {
        mi::Sint32 distill_errors = 0;
        mi::base::Handle<mi::neuraylib::ICompiled_material> distilled(
            distiller->distill_material(
                compiled.get(), "transmissive_pbr", nullptr, &distill_errors));
        if (distill_errors == 0 && distilled.is_valid_interface()) {
            PbrPaths paths;
            discover_distilled_paths(transaction.get(), distilled.get(), true, &paths);
            collect_sdk_textures(transaction.get(), distilled.get(), paths, out);
            wrote = apply_paths(distilled.get(), paths, out);
        }
        if (!wrote) {
            distill_errors = 0;
            distilled = mi::base::Handle<mi::neuraylib::ICompiled_material>(
                distiller->distill_material(
                    compiled.get(), "ue4", nullptr, &distill_errors));
            if (distill_errors == 0 && distilled.is_valid_interface()) {
                PbrPaths paths;
                discover_distilled_paths(transaction.get(), distilled.get(), false, &paths);
                collect_sdk_textures(transaction.get(), distilled.get(), paths, out);
                wrote = apply_paths(distilled.get(), paths, out);
            }
        }
    }

    if (!wrote) {
        PbrPaths paths;
        discover_distilled_paths(transaction.get(), compiled.get(), false, &paths);
        collect_sdk_textures(transaction.get(), compiled.get(), paths, out);
        wrote = apply_paths(compiled.get(), paths, out);
    }

    compiled = nullptr;
    material_instance = nullptr;
    material_type = nullptr;
    type_factory = nullptr;
    call = nullptr;
    override_arguments = nullptr;
    definition = nullptr;
    module = nullptr;
    module_db_name = nullptr;
    context = nullptr;
    transaction->commit();
    if (getenv("NUSD_MAT_DIAG") && wrote)
        fprintf(stderr,
                "[mat_diag] MDL SDK decoded material %s from %s "
                "base=%d(%.3f %.3f %.3f) rough=%d(%.3f) metal=%d(%.3f) "
                "opacity=%d(%.3f) transmission=%d(%.3f)\n",
                definition_db_name.c_str(), source_asset.c_str(),
                out->has_base_color, out->base_color[0], out->base_color[1],
                out->base_color[2],
                out->has_roughness, out->roughness,
                out->has_metallic, out->metallic,
                out->has_opacity, out->opacity,
                out->has_transmission_weight, out->transmission_weight);
    return wrote || out->texture_count > 0;
}

} // namespace

int nusd_mdl_bridge_available(void)
{
    std::lock_guard<std::mutex> lock(g_state.mutex);
    if (g_state.started) return 1;
    return getenv("NUSD_MDL_SDK_LIBRARY") || !g_state.load_attempted;
}

int nusd_mdl_bridge_decode(const char* mdl_source_asset,
                           const char* subidentifier,
                           const char* scene_dir,
                           NusdMdlDecoded* out)
{
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    for (int i = 0; i < 4; ++i) {
        out->base_color[i] = 1.0f;
        out->transmission_color[i] = 1.0f;
        out->specular_color[i] = 0.0f;
        out->emissive_color[i] = 0.0f;
    }
    out->opacity = 1.0f;
    out->ior = 1.5f;
    out->roughness = 0.5f;

    std::lock_guard<std::mutex> lock(g_state.mutex);
    return decode_with_sdk(mdl_source_asset, subidentifier, scene_dir,
                           nullptr, 0, out) ? 1 : 0;
}

int nusd_mdl_bridge_decode_with_inputs(const char* mdl_source_asset,
                                       const char* subidentifier,
                                       const char* scene_dir,
                                       const NusdMdlInput* inputs,
                                       int input_count,
                                       NusdMdlDecoded* out)
{
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    for (int i = 0; i < 4; ++i) {
        out->base_color[i] = 1.0f;
        out->transmission_color[i] = 1.0f;
        out->specular_color[i] = 0.0f;
        out->emissive_color[i] = 0.0f;
    }
    out->opacity = 1.0f;
    out->ior = 1.5f;
    out->roughness = 0.5f;

    std::lock_guard<std::mutex> lock(g_state.mutex);
    return decode_with_sdk(mdl_source_asset, subidentifier, scene_dir,
                           inputs, input_count, out) ? 1 : 0;
}

#endif
