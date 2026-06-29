// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NANOUSD_BUILDING
#define NANOUSD_BUILDING
#endif

#include "nanousd/nanousdapi.h"
#include "nanousd/nanousd_backend.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   include <windows.h>
#else
#   include <dlfcn.h>
#endif

// ============================================================
// Backend loader (singleton)
// ============================================================

static NanousdBackend_v1* g_backend = nullptr;

#ifdef _WIN32
static HMODULE g_backendLib = nullptr;
#else
static void* g_backendLib = nullptr;
#endif

static const char* DefaultBackendName() {
#ifdef _WIN32
    return "nanousd.dll";
#elif defined(__APPLE__)
    return "libnanousd.dylib";
#else
    return "libnanousd.so";
#endif
}

static bool LoadBackend() {
    if (g_backend) return true;

    // If a backend is already linked into the process (e.g. consumer links
    // both nanousdapi and nanousd), use it directly without dlopen.
    NanousdCreateBackendFn linkedFn = nullptr;
#ifdef _WIN32
    HMODULE exe = GetModuleHandleA(nullptr);
    if (exe)
        linkedFn = reinterpret_cast<NanousdCreateBackendFn>(
            GetProcAddress(exe, "nanousd_create_backend_v1"));
#else
    linkedFn = reinterpret_cast<NanousdCreateBackendFn>(
        dlsym(RTLD_DEFAULT, "nanousd_create_backend_v1"));
#endif
    if (linkedFn) {
        g_backend = linkedFn();
        return g_backend != nullptr;
    }

    // Otherwise, load the backend dynamically.
    const char* envPath = std::getenv("NANOUSD_BACKEND");
    std::string libPath;
    if (envPath && envPath[0]) {
        libPath = envPath;
    } else {
        // Look next to the nanousdapi library (or executable if statically linked)
#ifdef _WIN32
        HMODULE selfModule = nullptr;
        GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&LoadBackend),
            &selfModule);
        if (selfModule) {
            char buf[MAX_PATH];
            DWORD len = GetModuleFileNameA(selfModule, buf, MAX_PATH);
            if (len > 0) {
                std::string selfDir(buf, len);
                auto sep = selfDir.find_last_of("\\/");
                if (sep != std::string::npos) {
                    selfDir.resize(sep + 1);
                }
                libPath = selfDir + DefaultBackendName();
            }
        }
        if (libPath.empty()) {
            libPath = DefaultBackendName();
        }
#else
        Dl_info info;
        if (dladdr(reinterpret_cast<void*>(&LoadBackend), &info) && info.dli_fname) {
            std::string selfDir(info.dli_fname);
            auto sep = selfDir.find_last_of('/');
            if (sep != std::string::npos) {
                selfDir.resize(sep + 1);
            }
            libPath = selfDir + DefaultBackendName();
        } else {
            libPath = DefaultBackendName();
        }
#endif
    }

#ifdef _WIN32
    g_backendLib = LoadLibraryA(libPath.c_str());
    if (!g_backendLib) {
        fprintf(stderr, "nanousd: failed to load backend '%s' (error %lu)\n",
                libPath.c_str(), GetLastError());
        return false;
    }
    auto createFn = reinterpret_cast<NanousdCreateBackendFn>(
        GetProcAddress(g_backendLib, "nanousd_create_backend_v1"));
#else
    g_backendLib = dlopen(libPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!g_backendLib) {
        fprintf(stderr, "nanousd: failed to load backend '%s': %s\n",
                libPath.c_str(), dlerror());
        return false;
    }
    auto createFn = reinterpret_cast<NanousdCreateBackendFn>(
        dlsym(g_backendLib, "nanousd_create_backend_v1"));
#endif

    if (!createFn) {
        fprintf(stderr, "nanousd: backend '%s' missing entry point 'nanousd_create_backend_v1'\n",
                libPath.c_str());
        return false;
    }

    g_backend = createFn();
    if (!g_backend) {
        fprintf(stderr, "nanousd: backend '%s' returned null vtable\n", libPath.c_str());
        return false;
    }

    return true;
}

// Macro for dispatch — loads backend on first call, returns fallback on failure
#define ENSURE_BACKEND_OR(fallback) \
    do { if (!g_backend && !LoadBackend()) return fallback; } while (0)

// ============================================================
// Public C API — thin dispatch through backend vtable
// ============================================================

extern "C" {

NanousdStage nanousd_open(const char* filepath) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->open(filepath);
}

NanousdStage nanousd_open_masked(const char* filepath,
                                 const char* const* mask_paths,
                                 int mask_path_count) {
    ENSURE_BACKEND_OR(nullptr);
    if (!g_backend->open_masked) return nullptr;
    return g_backend->open_masked(filepath, mask_paths, mask_path_count);
}

void nanousd_close(NanousdStage stage) {
    if (!g_backend) return;
    g_backend->close(stage);
}

int nanousd_isvalid(NanousdStage stage) {
    ENSURE_BACKEND_OR(0);
    return g_backend->isvalid(stage);
}

const char* nanousd_error(NanousdStage stage) {
    ENSURE_BACKEND_OR("backend not loaded");
    return g_backend->error(stage);
}

const char* nanousd_stage_get_root_layer_path(NanousdStage stage) {
    ENSURE_BACKEND_OR("");
    return g_backend->stage_get_root_layer_path(stage);
}

int nanousd_stage_n_layers(NanousdStage stage) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->stage_n_layers) return 0;
    return g_backend->stage_n_layers(stage);
}

const char* nanousd_stage_layer_path(NanousdStage stage, int index) {
    ENSURE_BACKEND_OR("");
    if (!g_backend->stage_layer_path) return "";
    return g_backend->stage_layer_path(stage, index);
}

int nanousd_resolve_asset_path(const char* anchorLayerPath,
                               const char* assetPath,
                               char* out,
                               size_t out_size) {
    if (out && out_size > 0) out[0] = '\0';
    ENSURE_BACKEND_OR(0);
    if (!g_backend->resolve_asset_path) return 0;
    return g_backend->resolve_asset_path(anchorLayerPath, assetPath, out, out_size);
}

int nanousd_stage_resolve_asset_path(NanousdStage stage,
                                     const char* assetPath,
                                     char* out,
                                     size_t out_size) {
    if (out && out_size > 0) out[0] = '\0';
    ENSURE_BACKEND_OR(0);
    if (!g_backend->stage_resolve_asset_path) return 0;
    return g_backend->stage_resolve_asset_path(stage, assetPath, out, out_size);
}

int nanousd_read_asset_bytes(const char* resolvedLocation,
                             unsigned char** out_data,
                             size_t* out_size) {
    if (out_data) *out_data = nullptr;
    if (out_size) *out_size = 0;
    ENSURE_BACKEND_OR(0);
    if (!g_backend->read_asset_bytes) return 0;
    return g_backend->read_asset_bytes(resolvedLocation, out_data, out_size);
}

void nanousd_free_bytes(void* data) {
    if (!data) return;
    if (!g_backend && !LoadBackend()) {
        std::free(data);
        return;
    }
    if (g_backend->free_bytes) {
        g_backend->free_bytes(data);
        return;
    }
    std::free(data);
}

int nanousd_layer_has_prim_spec(NanousdStage stage, int layerIdx,
                                 const char* primPath) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->layer_has_prim_spec) return 0;
    return g_backend->layer_has_prim_spec(stage, layerIdx, primPath);
}

int nanousd_layer_has_attr_opinion(NanousdStage stage, int layerIdx,
                                    const char* primPath, const char* attrName) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->layer_has_attr_opinion) return 0;
    return g_backend->layer_has_attr_opinion(stage, layerIdx, primPath, attrName);
}

int nanousd_layer_attr_nsamples(NanousdStage stage, int layerIdx,
                                 const char* primPath, const char* attrName) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->layer_attr_nsamples) return 0;
    return g_backend->layer_attr_nsamples(stage, layerIdx, primPath, attrName);
}

NanousdListOp nanousd_layer_prim_listop(NanousdStage stage, int layerIdx,
                                        const char* primPath, const char* field) {
    ENSURE_BACKEND_OR(nullptr);
    if (!g_backend->layer_prim_listop) return nullptr;
    return g_backend->layer_prim_listop(stage, layerIdx, primPath, field);
}

int nanousd_layer_n_sublayers(NanousdStage stage, int layerIdx) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->layer_n_sublayers) return 0;
    return g_backend->layer_n_sublayers(stage, layerIdx);
}

const char* nanousd_layer_sublayer_path(NanousdStage stage, int layerIdx, int subIdx) {
    ENSURE_BACKEND_OR("");
    if (!g_backend->layer_sublayer_path) return "";
    return g_backend->layer_sublayer_path(stage, layerIdx, subIdx);
}

int nanousd_layer_offset(NanousdStage stage, int layerIdx,
                          double* offset, double* scale) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->layer_offset) return 0;
    return g_backend->layer_offset(stage, layerIdx, offset, scale);
}

double nanousd_timecodes_per_second(NanousdStage stage) {
    ENSURE_BACKEND_OR(24.0);
    return g_backend->timecodes_per_second(stage);
}

double nanousd_frames_per_second(NanousdStage stage) {
    ENSURE_BACKEND_OR(24.0);
    return g_backend->frames_per_second(stage);
}

double nanousd_start_time(NanousdStage stage) {
    ENSURE_BACKEND_OR(0.0);
    return g_backend->start_time(stage);
}

double nanousd_end_time(NanousdStage stage) {
    ENSURE_BACKEND_OR(0.0);
    return g_backend->end_time(stage);
}

int nanousd_nprims(NanousdStage stage) {
    ENSURE_BACKEND_OR(0);
    return g_backend->nprims(stage);
}

NanousdPrim nanousd_prim(NanousdStage stage, int index) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->prim(stage, index);
}

NanousdPrim nanousd_primpath(NanousdStage stage, const char* path) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->primpath(stage, path);
}

NanousdPrim nanousd_defaultprim(NanousdStage stage) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->defaultprim(stage);
}

int nanousd_traverse_flat(NanousdStage stage, NanousdFlatPrim* out,
                          int max_count) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->traverse_flat) return 0;
    return g_backend->traverse_flat(stage, out, max_count);
}

int nanousd_nchildren(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    return g_backend->nchildren(prim);
}

NanousdPrim nanousd_child(NanousdPrim prim, int index) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->child(prim, index);
}

NanousdPrim nanousd_childname(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->childname(prim, name);
}

const char* nanousd_path(NanousdPrim prim) {
    ENSURE_BACKEND_OR("");
    return g_backend->path(prim);
}

const char* nanousd_name(NanousdPrim prim) {
    ENSURE_BACKEND_OR("");
    return g_backend->name(prim);
}

const char* nanousd_typename(NanousdPrim prim) {
    ENSURE_BACKEND_OR("");
    return g_backend->type_name(prim);
}

const char* nanousd_kind(NanousdPrim prim) {
    ENSURE_BACKEND_OR("");
    return g_backend->kind(prim);
}

int nanousd_isactive(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    return g_backend->isactive(prim);
}

int nanousd_isdefined(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    return g_backend->isdefined(prim);
}

int nanousd_isabstract(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    return g_backend->isabstract(prim);
}

int nanousd_isinstanceable(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    return g_backend->isinstanceable(prim);
}

int nanousd_prim_isvalid(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    return g_backend->prim_isvalid(prim);
}

int nanousd_nattribs(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    return g_backend->nattribs(prim);
}

const char* nanousd_attribname(NanousdPrim prim, int index) {
    ENSURE_BACKEND_OR("");
    return g_backend->attribname(prim, index);
}

int nanousd_nauthored_attribs(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->nauthored_attribs) return 0;
    return g_backend->nauthored_attribs(prim);
}

const char* nanousd_authored_attribname(NanousdPrim prim, int index) {
    ENSURE_BACKEND_OR("");
    if (!g_backend->authored_attribname) return "";
    return g_backend->authored_attribname(prim, index);
}

int nanousd_hasattrib(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    return g_backend->hasattrib(prim, name);
}

const char* nanousd_attribtype(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR("");
    return g_backend->attribtype(prim, name);
}

int nanousd_nproperties(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->nproperties) return 0;
    return g_backend->nproperties(prim);
}

const char* nanousd_propertyname(NanousdPrim prim, int index) {
    ENSURE_BACKEND_OR("");
    if (!g_backend->propertyname) return "";
    return g_backend->propertyname(prim, index);
}

int nanousd_property_is_attribute(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->property_is_attribute) return nanousd_hasattrib(prim, name);
    return g_backend->property_is_attribute(prim, name);
}

int nanousd_property_is_relationship(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->property_is_relationship) return nanousd_hasrel(prim, name);
    return g_backend->property_is_relationship(prim, name);
}

float nanousd_attribf(NanousdPrim prim, const char* name, int* ok) {
    ENSURE_BACKEND_OR(0.0f);
    return g_backend->attribf(prim, name, ok);
}

double nanousd_attribd(NanousdPrim prim, const char* name, int* ok) {
    ENSURE_BACKEND_OR(0.0);
    return g_backend->attribd(prim, name, ok);
}

int nanousd_attribi(NanousdPrim prim, const char* name, int* ok) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribi(prim, name, ok);
}

const char* nanousd_attribs(NanousdPrim prim, const char* name, int* ok) {
    ENSURE_BACKEND_OR("");
    return g_backend->attribs(prim, name, ok);
}

int nanousd_attribv2f(NanousdPrim prim, const char* name, float out[2]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribv2f(prim, name, out);
}

int nanousd_attribv3f(NanousdPrim prim, const char* name, float out[3]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribv3f(prim, name, out);
}

int nanousd_attribv4f(NanousdPrim prim, const char* name, float out[4]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribv4f(prim, name, out);
}

int nanousd_attribv2d(NanousdPrim prim, const char* name, double out[2]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribv2d(prim, name, out);
}

int nanousd_attribv3d(NanousdPrim prim, const char* name, double out[3]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribv3d(prim, name, out);
}

int nanousd_attribv4d(NanousdPrim prim, const char* name, double out[4]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribv4d(prim, name, out);
}

int nanousd_attribv2i(NanousdPrim prim, const char* name, int out[2]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribv2i(prim, name, out);
}

int nanousd_attribv3i(NanousdPrim prim, const char* name, int out[3]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribv3i(prim, name, out);
}

int nanousd_attribv4i(NanousdPrim prim, const char* name, int out[4]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribv4i(prim, name, out);
}

int nanousd_attribm4d(NanousdPrim prim, const char* name, double out[16]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribm4d(prim, name, out);
}

int nanousd_attribarraylen(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(-1);
    return g_backend->attribarraylen(prim, name);
}

int nanousd_attribarrayf(NanousdPrim prim, const char* name, float* out, int maxlen) {
    ENSURE_BACKEND_OR(-1);
    return g_backend->attribarrayf(prim, name, out, maxlen);
}

int nanousd_attribarrayi(NanousdPrim prim, const char* name, int* out, int maxlen) {
    ENSURE_BACKEND_OR(-1);
    return g_backend->attribarrayi(prim, name, out, maxlen);
}

int nanousd_hassamples(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    return g_backend->hassamples(prim, name);
}

int nanousd_nsamplekeys(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    return g_backend->nsamplekeys(prim, name);
}

double nanousd_samplekey(NanousdPrim prim, const char* name, int index) {
    ENSURE_BACKEND_OR(0.0);
    return g_backend->samplekey(prim, name, index);
}

float nanousd_samplef(NanousdPrim prim, const char* name, double time, int* ok) {
    ENSURE_BACKEND_OR(0.0f);
    return g_backend->samplef(prim, name, time, ok);
}

int nanousd_samplev3f(NanousdPrim prim, const char* name, double time, float out[3]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->samplev3f(prim, name, time, out);
}

int nanousd_samplev3d(NanousdPrim prim, const char* name, double time, double out[3]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->samplev3d(prim, name, time, out);
}

int nanousd_hasrel(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    return g_backend->hasrel(prim, name);
}

int nanousd_rel_authored(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->rel_authored) return nanousd_hasrel(prim, name);
    return g_backend->rel_authored(prim, name);
}

void nanousd_freeprim(NanousdPrim prim) {
    if (!g_backend) return;
    g_backend->freeprim(prim);
}

// ============================================================
// Extensions (appended to preserve v1 ABI)
// ============================================================

double nanousd_metadatad(NanousdStage stage, const char* key, int* ok) {
    ENSURE_BACKEND_OR(0.0);
    return g_backend->metadatad(stage, key, ok);
}

const char* nanousd_metadatas(NanousdStage stage, const char* key, int* ok) {
    ENSURE_BACKEND_OR("");
    return g_backend->metadatas(stage, key, ok);
}

int nanousd_set_stage_metadatad(NanousdStage stage, const char* key, double value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_stage_metadatad(stage, key, value);
}

int nanousd_set_stage_metadatas(NanousdStage stage, const char* key, const char* value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_stage_metadatas(stage, key, value);
}

int nanousd_set_stage_metadata_token(NanousdStage stage, const char* key, const char* value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_stage_metadata_token(stage, key, value);
}

int nanousd_isa(NanousdPrim prim, const char* typeName) {
    ENSURE_BACKEND_OR(0);
    return g_backend->isa(prim, typeName);
}

int nanousd_hasapi(NanousdPrim prim, const char* apiName) {
    ENSURE_BACKEND_OR(0);
    return g_backend->hasapi(prim, apiName);
}

int64_t nanousd_attribi64(NanousdPrim prim, const char* name, int* ok) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribi64(prim, name, ok);
}

int nanousd_attribb(NanousdPrim prim, const char* name, int* ok) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribb(prim, name, ok);
}

int nanousd_attribarrayd(NanousdPrim prim, const char* name, double* out, int maxlen) {
    ENSURE_BACKEND_OR(-1);
    return g_backend->attribarrayd(prim, name, out, maxlen);
}

double nanousd_sampled(NanousdPrim prim, const char* name, double time, int* ok) {
    ENSURE_BACKEND_OR(0.0);
    return g_backend->sampled(prim, name, time, ok);
}

int nanousd_nreltargets(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    return g_backend->nreltargets(prim, name);
}

const char* nanousd_reltarget(NanousdPrim prim, const char* name, int index) {
    ENSURE_BACKEND_OR("");
    return g_backend->reltarget(prim, name, index);
}

const char* nanousd_rel_metadatas(NanousdPrim prim, const char* relName,
                                  const char* key, int* ok) {
    ENSURE_BACKEND_OR("");
    if (!g_backend->rel_metadatas) {
        if (ok) *ok = 0;
        return "";
    }
    return g_backend->rel_metadatas(prim, relName, key, ok);
}

int nanousd_collection_nmembers(NanousdPrim prim, const char* instance_name) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->collection_nmembers) return 0;
    return g_backend->collection_nmembers(prim, instance_name);
}

const char* nanousd_collection_member(NanousdPrim prim,
                                      const char* instance_name,
                                      int index) {
    ENSURE_BACKEND_OR("");
    if (!g_backend->collection_member) return "";
    return g_backend->collection_member(prim, instance_name, index);
}

int nanousd_collection_contains(NanousdPrim prim,
                                const char* instance_name,
                                const char* path) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->collection_contains) return 0;
    return g_backend->collection_contains(prim, instance_name, path);
}

// --- Paths ---

NanousdPath nanousd_path_parse(const char* text) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->path_parse(text);
}

const char* nanousd_path_str(NanousdPath path) {
    ENSURE_BACKEND_OR("");
    return g_backend->path_str(path);
}

NanousdPath nanousd_path_append_child(NanousdPath parent, const char* child) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->path_append_child(parent, child);
}

NanousdPath nanousd_path_append_property(NanousdPath prim, const char* prop) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->path_append_property(prim, prop);
}

NanousdPath nanousd_path_parent(NanousdPath path) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->path_parent(path);
}

const char* nanousd_path_name(NanousdPath path) {
    ENSURE_BACKEND_OR("");
    return g_backend->path_name(path);
}

int nanousd_path_is_absolute(NanousdPath path) {
    ENSURE_BACKEND_OR(0);
    return g_backend->path_is_absolute(path);
}

int nanousd_path_is_root(NanousdPath path) {
    ENSURE_BACKEND_OR(0);
    return g_backend->path_is_root(path);
}

int nanousd_path_is_property(NanousdPath path) {
    ENSURE_BACKEND_OR(0);
    return g_backend->path_is_property(path);
}

int nanousd_path_equal(NanousdPath a, NanousdPath b) {
    ENSURE_BACKEND_OR(0);
    return g_backend->path_equal(a, b);
}

void nanousd_path_free(NanousdPath path) {
    if (!g_backend) return;
    g_backend->path_free(path);
}

// --- ListOps ---

NanousdListOp nanousd_listop_create_explicit(const char** items, int count) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->listop_create_explicit(items, count);
}

NanousdListOp nanousd_listop_create(const char** prepend, int nprepend,
                                 const char** append, int nappend,
                                 const char** delete_, int ndelete) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->listop_create(prepend, nprepend, append, nappend, delete_, ndelete);
}

void nanousd_listop_free(NanousdListOp op) {
    if (!g_backend) return;
    g_backend->listop_free(op);
}

int nanousd_listop_is_explicit(NanousdListOp op) {
    ENSURE_BACKEND_OR(0);
    return g_backend->listop_is_explicit(op);
}

int nanousd_listop_nitems(NanousdListOp op) {
    ENSURE_BACKEND_OR(0);
    return g_backend->listop_nitems(op);
}

const char* nanousd_listop_item(NanousdListOp op, int index) {
    ENSURE_BACKEND_OR("");
    return g_backend->listop_item(op, index);
}

int nanousd_listop_nprepended(NanousdListOp op) {
    ENSURE_BACKEND_OR(0);
    return g_backend->listop_nprepended(op);
}

const char* nanousd_listop_prepended(NanousdListOp op, int index) {
    ENSURE_BACKEND_OR("");
    return g_backend->listop_prepended(op, index);
}

int nanousd_listop_nappended(NanousdListOp op) {
    ENSURE_BACKEND_OR(0);
    return g_backend->listop_nappended(op);
}

const char* nanousd_listop_appended(NanousdListOp op, int index) {
    ENSURE_BACKEND_OR("");
    return g_backend->listop_appended(op, index);
}

int nanousd_listop_ndeleted(NanousdListOp op) {
    ENSURE_BACKEND_OR(0);
    return g_backend->listop_ndeleted(op);
}

const char* nanousd_listop_deleted(NanousdListOp op, int index) {
    ENSURE_BACKEND_OR("");
    return g_backend->listop_deleted(op, index);
}

NanousdListOp nanousd_listop_combine(NanousdListOp stronger, NanousdListOp weaker) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->listop_combine(stronger, weaker);
}

NanousdListOp nanousd_prim_listop(NanousdPrim prim, const char* field) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->prim_listop(prim, field);
}

// --- Vec/Matrix/Quaternion utilities ---

float nanousd_dot3f(const float a[3], const float b[3]) {
    ENSURE_BACKEND_OR(0.0f);
    return g_backend->dot3f(a, b);
}

double nanousd_dot3d(const double a[3], const double b[3]) {
    ENSURE_BACKEND_OR(0.0);
    return g_backend->dot3d(a, b);
}

float nanousd_length3f(const float v[3]) {
    ENSURE_BACKEND_OR(0.0f);
    return g_backend->length3f(v);
}

double nanousd_length3d(const double v[3]) {
    ENSURE_BACKEND_OR(0.0);
    return g_backend->length3d(v);
}

void nanousd_normalize3f(const float v[3], float out[3]) {
    if (!g_backend && !LoadBackend()) return;
    g_backend->normalize3f(v, out);
}

void nanousd_normalize3d(const double v[3], double out[3]) {
    if (!g_backend && !LoadBackend()) return;
    g_backend->normalize3d(v, out);
}

void nanousd_cross3f(const float a[3], const float b[3], float out[3]) {
    if (!g_backend && !LoadBackend()) return;
    g_backend->cross3f(a, b, out);
}

void nanousd_cross3d(const double a[3], const double b[3], double out[3]) {
    if (!g_backend && !LoadBackend()) return;
    g_backend->cross3d(a, b, out);
}

void nanousd_mul_m4d(const double a[16], const double b[16], double out[16]) {
    if (!g_backend && !LoadBackend()) return;
    g_backend->mul_m4d(a, b, out);
}

void nanousd_transform_point3d(const double m[16], const double p[3], double out[3]) {
    if (!g_backend && !LoadBackend()) return;
    g_backend->transform_point3d(m, p, out);
}

void nanousd_quat_slerp(const double a[4], const double b[4], double t, double out[4]) {
    if (!g_backend && !LoadBackend()) return;
    g_backend->quat_slerp(a, b, t, out);
}

void nanousd_quat_to_matrix(const double q[4], double out[16]) {
    if (!g_backend && !LoadBackend()) return;
    g_backend->quat_to_matrix(q, out);
}

// ============================================================
// Attribute write operations
// ============================================================

int nanousd_set_attribf(NanousdPrim prim, const char* name, float value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribf(prim, name, value);
}

int nanousd_set_attribd(NanousdPrim prim, const char* name, double value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribd(prim, name, value);
}

int nanousd_set_attribi(NanousdPrim prim, const char* name, int value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribi(prim, name, value);
}

int nanousd_set_attribs(NanousdPrim prim, const char* name, const char* value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribs(prim, name, value);
}

int nanousd_set_attribb(NanousdPrim prim, const char* name, int value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribb(prim, name, value);
}

int nanousd_set_attribi64(NanousdPrim prim, const char* name, int64_t value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribi64(prim, name, value);
}

int nanousd_set_attribv2f(NanousdPrim prim, const char* name, const float v[2]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribv2f(prim, name, v);
}

int nanousd_set_attribv3f(NanousdPrim prim, const char* name, const float v[3]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribv3f(prim, name, v);
}

int nanousd_set_attribv4f(NanousdPrim prim, const char* name, const float v[4]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribv4f(prim, name, v);
}

int nanousd_set_attribv2d(NanousdPrim prim, const char* name, const double v[2]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribv2d(prim, name, v);
}

int nanousd_set_attribv3d(NanousdPrim prim, const char* name, const double v[3]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribv3d(prim, name, v);
}

int nanousd_set_attribv4d(NanousdPrim prim, const char* name, const double v[4]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribv4d(prim, name, v);
}

int nanousd_set_attribv2i(NanousdPrim prim, const char* name, const int v[2]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribv2i(prim, name, v);
}

int nanousd_set_attribv3i(NanousdPrim prim, const char* name, const int v[3]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribv3i(prim, name, v);
}

int nanousd_set_attribv4i(NanousdPrim prim, const char* name, const int v[4]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribv4i(prim, name, v);
}

int nanousd_set_attribm4d(NanousdPrim prim, const char* name, const double v[16]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribm4d(prim, name, v);
}

int nanousd_set_attribarrayf(NanousdPrim prim, const char* name, const float* data, int count) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribarrayf(prim, name, data, count);
}

int nanousd_set_attribarrayd(NanousdPrim prim, const char* name, const double* data, int count) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribarrayd(prim, name, data, count);
}

int nanousd_set_attribarrayi(NanousdPrim prim, const char* name, const int* data, int count) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribarrayi(prim, name, data, count);
}

int nanousd_set_samplef(NanousdPrim prim, const char* name, double time, float value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_samplef(prim, name, time, value);
}

int nanousd_set_sampled(NanousdPrim prim, const char* name, double time, double value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_sampled(prim, name, time, value);
}

int nanousd_set_samplev3f(NanousdPrim prim, const char* name, double time, const float v[3]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_samplev3f(prim, name, time, v);
}

int nanousd_set_samplev3d(NanousdPrim prim, const char* name, double time, const double v[3]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_samplev3d(prim, name, time, v);
}

int nanousd_set_samplev4f(NanousdPrim prim, const char* name, double time, const float v[4]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_samplev4f(prim, name, time, v);
}

int nanousd_set_sampleqf(NanousdPrim prim, const char* name, double time, const float v[4]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_sampleqf(prim, name, time, v);
}

int nanousd_set_sample_token(NanousdPrim prim, const char* name, double time, const char* value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_sample_token(prim, name, time, value);
}

int nanousd_set_samplearrayf(NanousdPrim prim, const char* name, double time, const float* data, int count) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_samplearrayf(prim, name, time, data, count);
}

int nanousd_set_samplearrayi(NanousdPrim prim, const char* name, double time, const int* data, int count) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_samplearrayi(prim, name, time, data, count);
}

int nanousd_set_samplearrayv3f(NanousdPrim prim, const char* name, double time, const float* data, int count) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_samplearrayv3f(prim, name, time, data, count);
}

int nanousd_set_samplev2d(NanousdPrim prim, const char* name, double time, const double v[2]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_samplev2d(prim, name, time, v);
}

int nanousd_set_samplev4d(NanousdPrim prim, const char* name, double time, const double v[4]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_samplev4d(prim, name, time, v);
}

int nanousd_set_samplem4d(NanousdPrim prim, const char* name, double time, const double v[16]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_samplem4d(prim, name, time, v);
}

int nanousd_set_samplearrayd(NanousdPrim prim, const char* name, double time, const double* data, int count) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_samplearrayd(prim, name, time, data, count);
}

int nanousd_set_samplearrayv3d(NanousdPrim prim, const char* name, double time, const double* data, int count) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_samplearrayv3d(prim, name, time, data, count);
}

int nanousd_clear_default(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    return g_backend->clear_default(prim, name);
}

int nanousd_clear_samples(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    return g_backend->clear_samples(prim, name);
}

int nanousd_block_attrib(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    return g_backend->block_attrib(prim, name);
}

int nanousd_create_attrib(NanousdPrim prim, const char* name, const char* typeName) {
    ENSURE_BACKEND_OR(0);
    return g_backend->create_attrib(prim, name, typeName);
}

// ============================================================
// Bulk array access (GPU-friendly)
// ============================================================

const float* nanousd_arraydataf(NanousdPrim prim, const char* name, int* count) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->arraydataf(prim, name, count);
}

const double* nanousd_arraydatad(NanousdPrim prim, const char* name, int* count) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->arraydatad(prim, name, count);
}

const int* nanousd_arraydatai(NanousdPrim prim, const char* name, int* count) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->arraydatai(prim, name, count);
}

int nanousd_attribarrayv3f(NanousdPrim prim, const char* name, float* out, int maxcount) {
    ENSURE_BACKEND_OR(-1);
    return g_backend->attribarrayv3f(prim, name, out, maxcount);
}

int nanousd_attribarrayv3d(NanousdPrim prim, const char* name, double* out, int maxcount) {
    ENSURE_BACKEND_OR(-1);
    return g_backend->attribarrayv3d(prim, name, out, maxcount);
}

int nanousd_set_attribarrayv3f(NanousdPrim prim, const char* name, const float* data, int count) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribarrayv3f(prim, name, data, count);
}

int nanousd_set_attribarrayv3d(NanousdPrim prim, const char* name, const double* data, int count) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribarrayv3d(prim, name, data, count);
}

// ============================================================
// Quaternion read/write
// ============================================================

int nanousd_attribqf(NanousdPrim prim, const char* name, float out[4]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribqf(prim, name, out);
}

int nanousd_attribqd(NanousdPrim prim, const char* name, double out[4]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribqd(prim, name, out);
}

int nanousd_set_attribqf(NanousdPrim prim, const char* name, const float v[4]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribqf(prim, name, v);
}

int nanousd_set_attribqd(NanousdPrim prim, const char* name, const double v[4]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribqd(prim, name, v);
}

// ============================================================
// Relationship write
// ============================================================

int nanousd_create_rel(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    return g_backend->create_rel(prim, name);
}

int nanousd_set_reltargets(NanousdPrim prim, const char* name,
                          const char** targets, int count) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_reltargets(prim, name, targets, count);
}

int nanousd_add_reltarget(NanousdPrim prim, const char* name, const char* target) {
    ENSURE_BACKEND_OR(0);
    return g_backend->add_reltarget(prim, name, target);
}

int nanousd_clear_reltargets(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    return g_backend->clear_reltargets(prim, name);
}

// ============================================================
// Stage creation
// ============================================================

NanousdStage nanousd_create(void) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->create();
}

// ============================================================
// Prim creation
// ============================================================

NanousdPrim nanousd_define_prim(NanousdStage stage, const char* path, const char* typeName) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->define_prim(stage, path, typeName);
}

NanousdPrim nanousd_define_prim_s(NanousdStage stage, const char* path,
                               const char* typeName, const char* specifier) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->define_prim_s(stage, path, typeName, specifier);
}

int nanousd_set_specifier(NanousdPrim prim, const char* specifier) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_specifier(prim, specifier);
}

// ============================================================
// Schema application
// ============================================================

int nanousd_apply_api(NanousdPrim prim, const char* schemaName) {
    ENSURE_BACKEND_OR(0);
    return g_backend->apply_api(prim, schemaName);
}

// --- P1 extensions ---

int nanousd_attribm3d(NanousdPrim prim, const char* name, double out[9]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attribm3d(prim, name, out);
}

int nanousd_set_attribm3d(NanousdPrim prim, const char* name, const double v[9]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribm3d(prim, name, v);
}

int nanousd_attribarrays_len(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(-1);
    return g_backend->attribarrays_len(prim, name);
}

const char* nanousd_attribarrays(NanousdPrim prim, const char* name, int index) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->attribarrays(prim, name, index);
}

const char* nanousd_attribarrays_elem(NanousdPrim prim, const char* name, int index) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->attribarrays(prim, name, index);
}

int nanousd_set_attribarrays(NanousdPrim prim, const char* name,
                            const char** strings, int count) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribarrays(prim, name, strings, count);
}

const char* nanousd_attribasset(NanousdPrim prim, const char* name, int* ok) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->attribasset(prim, name, ok);
}

int nanousd_get_local_transform(NanousdPrim prim, double time, double out[16],
                               int* resetXformStack) {
    ENSURE_BACKEND_OR(0);
    return g_backend->get_local_transform(prim, time, out, resetXformStack);
}

int nanousd_write_usdc(NanousdStage stage, const char* filepath) {
    ENSURE_BACKEND_OR(0);
    return g_backend->write_usdc(stage, filepath);
}

int nanousd_write_usdz(NanousdStage stage, const char* filepath) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->write_usdz) return 0;
    return g_backend->write_usdz(stage, filepath);
}

int nanousd_write_usda(NanousdStage stage, const char* filepath) {
    ENSURE_BACKEND_OR(0);
    return g_backend->write_usda(stage, filepath);
}

const char* nanousd_write_usda_string(NanousdStage stage) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->write_usda_string(stage);
}

void nanousd_free_string(const char* str) {
    free(const_cast<char*>(str));
}

// --- Array time sample reads ---

int nanousd_samplev2f(NanousdPrim prim, const char* name, double time, float out[2]) {
    ENSURE_BACKEND_OR(0);
    return g_backend->samplev2f(prim, name, time, out);
}

int nanousd_samplearrayf(NanousdPrim prim, const char* name, double time,
                        float* out, int maxlen) {
    ENSURE_BACKEND_OR(-1);
    return g_backend->samplearrayf(prim, name, time, out, maxlen);
}

int nanousd_samplearrayd(NanousdPrim prim, const char* name, double time,
                        double* out, int maxlen) {
    ENSURE_BACKEND_OR(-1);
    return g_backend->samplearrayd(prim, name, time, out, maxlen);
}

int nanousd_samplearrayi(NanousdPrim prim, const char* name, double time,
                        int* out, int maxlen) {
    ENSURE_BACKEND_OR(-1);
    return g_backend->samplearrayi(prim, name, time, out, maxlen);
}

const char* nanousd_attrib_token(NanousdPrim prim, const char* name, int* ok) {
    ENSURE_BACKEND_OR("");
    return g_backend->attrib_token(prim, name, ok);
}

int nanousd_set_attrib_token(NanousdPrim prim, const char* name, const char* value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attrib_token(prim, name, value);
}

int nanousd_set_attrib_asset(NanousdPrim prim, const char* name, const char* value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attrib_asset(prim, name, value);
}

int nanousd_set_attribarraytokens(NanousdPrim prim, const char* name,
                                 const char** values, int count) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_attribarraytokens(prim, name, values, count);
}

int nanousd_attribarraytokens_len(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(-1);
    return g_backend->attribarraytokens_len(prim, name);
}

const char* nanousd_attribarraytokens(NanousdPrim prim, const char* name, int index) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->attribarraytokens(prim, name, index);
}

int nanousd_add_reference(NanousdPrim prim, const char* assetPath,
                         const char* primPath) {
    ENSURE_BACKEND_OR(0);
    return g_backend->add_reference(prim, assetPath, primPath);
}

int nanousd_register_schemas_json(const char* json) {
    ENSURE_BACKEND_OR(0);
    return g_backend->register_schemas_json(json);
}

const char* nanousd_prim_metadatas(NanousdPrim prim, const char* key, int* ok) {
    ENSURE_BACKEND_OR("");
    return g_backend->prim_metadatas(prim, key, ok);
}

double nanousd_prim_metadatad(NanousdPrim prim, const char* key, int* ok) {
    ENSURE_BACKEND_OR(0.0);
    return g_backend->prim_metadatad(prim, key, ok);
}

int nanousd_set_prim_metadatas(NanousdPrim prim, const char* key, const char* value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_prim_metadatas(prim, key, value);
}

int nanousd_set_prim_metadatad(NanousdPrim prim, const char* key, double value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_prim_metadatad(prim, key, value);
}

int nanousd_set_prim_metadata_token(NanousdPrim prim, const char* key, const char* value) {
    ENSURE_BACKEND_OR(0);
    return g_backend->set_prim_metadata_token(prim, key, value);
}

// --- Attribute metadata & connections ---

const char* nanousd_attrib_interpolation(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->attrib_interpolation(prim, name);
}

int nanousd_attrib_authored(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    return g_backend->attrib_authored(prim, name);
}

const char* nanousd_attrib_colorspace(NanousdPrim prim, const char* name, int* ok) {
    if (ok) *ok = 0;
    ENSURE_BACKEND_OR("");
    if (!g_backend->attrib_colorspace) return "";
    return g_backend->attrib_colorspace(prim, name, ok);
}

const char* nanousd_attrib_resolved_colorspace(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR("");
    if (!g_backend->attrib_resolved_colorspace) return "";
    return g_backend->attrib_resolved_colorspace(prim, name);
}

int nanousd_set_attrib_colorspace(NanousdPrim prim, const char* name,
                                  const char* colorSpace) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->set_attrib_colorspace) return 0;
    return g_backend->set_attrib_colorspace(prim, name, colorSpace);
}

int nanousd_clear_attrib_colorspace(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->clear_attrib_colorspace) return 0;
    return g_backend->clear_attrib_colorspace(prim, name);
}

const char* nanousd_prim_resolved_colorspace(NanousdPrim prim) {
    ENSURE_BACKEND_OR("");
    if (!g_backend->prim_resolved_colorspace) return "";
    return g_backend->prim_resolved_colorspace(prim);
}

int nanousd_hasconnections(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    return g_backend->hasconnections(prim, name);
}

int nanousd_nconnections(NanousdPrim prim, const char* name) {
    ENSURE_BACKEND_OR(0);
    return g_backend->nconnections(prim, name);
}

const char* nanousd_connection(NanousdPrim prim, const char* name, int index) {
    ENSURE_BACKEND_OR("");
    return g_backend->connection(prim, name, index);
}

NanousdPrim nanousd_parent(NanousdPrim prim) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->parent(prim);
}

int nanousd_attribarrayi64(NanousdPrim prim, const char* name, int64_t* out, int maxlen) {
    ENSURE_BACKEND_OR(-1);
    return g_backend->attribarrayi64(prim, name, out, maxlen);
}

// --- Instancing ---

int nanousd_stage_nprototypes(NanousdStage stage) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->stage_nprototypes) return 0;
    return g_backend->stage_nprototypes(stage);
}

NanousdPrim nanousd_stage_prototype(NanousdStage stage, int index) {
    ENSURE_BACKEND_OR(nullptr);
    if (!g_backend->stage_prototype) return nullptr;
    return g_backend->stage_prototype(stage, index);
}

int nanousd_isinstance(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    return g_backend->isinstance(prim);
}

int nanousd_isprototype(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    return g_backend->isprototype(prim);
}

int nanousd_isinprototype(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    return g_backend->isinprototype(prim);
}

int nanousd_isinstanceproxy(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->isinstanceproxy) return 0;
    return g_backend->isinstanceproxy(prim);
}

NanousdPrim nanousd_prototype(NanousdPrim prim) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->prototype(prim);
}

NanousdPrim nanousd_priminprototype(NanousdPrim prim) {
    ENSURE_BACKEND_OR(nullptr);
    if (!g_backend->priminprototype) return nullptr;
    return g_backend->priminprototype(prim);
}

int nanousd_ninstances(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    return g_backend->ninstances(prim);
}

NanousdPrim nanousd_instance(NanousdPrim prim, int index) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->instance(prim, index);
}

int nanousd_instance_key(NanousdPrim prim, char* out, size_t out_size) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->instance_key) {
        if (out && out_size > 0) out[0] = '\0';
        return 0;
    }
    return g_backend->instance_key(prim, out, out_size);
}

int nanousd_ncomposition_arcs(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->ncomposition_arcs) return 0;
    return g_backend->ncomposition_arcs(prim);
}

int nanousd_composition_arc(NanousdPrim prim, int index,
                            NanousdCompositionArc* out) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->composition_arc) return 0;
    return g_backend->composition_arc(prim, index, out);
}

// --- Diagnostics ---

NanousdDiagnostic* nanousd_diagnostics(NanousdStage stage, int* count) {
    if (count) *count = 0;
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->diagnostics(stage, count);
}

void nanousd_free_diagnostics(NanousdDiagnostic* diags, int count) {
    if (!g_backend) return;
    g_backend->free_diagnostics(diags, count);
}

const char* nanousd_diagnostics_json(NanousdStage stage) {
    ENSURE_BACKEND_OR(nullptr);
    return g_backend->diagnostics_json(stage);
}

// --- Variants ---

int nanousd_nvariantsets(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    return g_backend->nvariantsets(prim);
}

const char* nanousd_variantsetname(NanousdPrim prim, int index) {
    ENSURE_BACKEND_OR("");
    return g_backend->variantsetname(prim, index);
}

int nanousd_hasvariantset(NanousdPrim prim, const char* setName) {
    ENSURE_BACKEND_OR(0);
    return g_backend->hasvariantset(prim, setName);
}

int nanousd_nvariants(NanousdPrim prim, const char* setName) {
    ENSURE_BACKEND_OR(0);
    return g_backend->nvariants(prim, setName);
}

const char* nanousd_variantname(NanousdPrim prim, const char* setName, int index) {
    ENSURE_BACKEND_OR("");
    return g_backend->variantname(prim, setName, index);
}

const char* nanousd_variantselection(NanousdPrim prim, const char* setName) {
    ENSURE_BACKEND_OR("");
    return g_backend->variantselection(prim, setName);
}

int nanousd_setvariantselection(NanousdPrim prim, const char* setName,
                                  const char* variantName, int layerIndex) {
    ENSURE_BACKEND_OR(0);
    return g_backend->setvariantselection(prim, setName, variantName, layerIndex);
}

// --- Composition-arc authoring (panel-c-api branch additions) ---

int nanousd_add_payload(NanousdPrim prim, const char* assetPath,
                         const char* primPath) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->add_payload) return 0;
    return g_backend->add_payload(prim, assetPath, primPath);
}

int nanousd_add_inherit(NanousdPrim prim, const char* primPath) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->add_inherit) return 0;
    return g_backend->add_inherit(prim, primPath);
}

int nanousd_add_specialize(NanousdPrim prim, const char* primPath) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->add_specialize) return 0;
    return g_backend->add_specialize(prim, primPath);
}

int nanousd_remove_listop_item(NanousdPrim prim, const char* field,
                                int listOpKind, int index) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->remove_listop_item) return 0;
    return g_backend->remove_listop_item(prim, field, listOpKind, index);
}

int nanousd_remove_reference(NanousdPrim prim, int index) {
    return nanousd_remove_listop_item(prim, "references", 1, index);
}
int nanousd_remove_payload(NanousdPrim prim, int index) {
    return nanousd_remove_listop_item(prim, "payload", 1, index);
}
int nanousd_remove_inherit(NanousdPrim prim, int index) {
    return nanousd_remove_listop_item(prim, "inheritPaths", 1, index);
}
int nanousd_remove_specialize(NanousdPrim prim, int index) {
    return nanousd_remove_listop_item(prim, "specializes", 1, index);
}

// --- Prim-state writers ---

int nanousd_set_active(NanousdPrim prim, int active) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->set_active) return 0;
    return g_backend->set_active(prim, active);
}

int nanousd_set_instanceable(NanousdPrim prim, int instanceable) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->set_instanceable) return 0;
    return g_backend->set_instanceable(prim, instanceable);
}

int nanousd_remove_api(NanousdPrim prim, const char* schemaName) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->remove_api) return 0;
    return g_backend->remove_api(prim, schemaName);
}

int nanousd_remove_prim(NanousdPrim prim) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->remove_prim) return 0;
    return g_backend->remove_prim(prim);
}

// --- Variant set authoring ---

int nanousd_create_variantset(NanousdPrim prim, const char* setName) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->create_variantset) return 0;
    return g_backend->create_variantset(prim, setName);
}

int nanousd_create_variant(NanousdPrim prim, const char* setName,
                            const char* variantName) {
    ENSURE_BACKEND_OR(0);
    if (!g_backend->create_variant) return 0;
    return g_backend->create_variant(prim, setName, variantName);
}

} // extern "C"
