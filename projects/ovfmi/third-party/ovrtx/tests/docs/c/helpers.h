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

#include <ovrtx/ovrtx.h>
#include <ovrtx/ovrtx_config.h>
#include <ovrtx/ovrtx_types.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "stb_image_write.h"
#include <ovx/path_dictionary/path_dictionary_utils.h>

// Return the path to tests/data/. Prefer the OVRTX_TEST_DATA_DIR environment
// variable (set by CI when build and test run on different machines) and fall
// back to a path relative to this source file for local development.
static std::string get_test_data_dir() {
    if (const char* env = std::getenv("OVRTX_TEST_DATA_DIR")) {
        return env;
    }
    std::filesystem::path src(__FILE__);
    return (src.parent_path() / "../../../tests/data").lexically_normal().string();
}

// Return the path to tests/docs/data/. Prefer the OVRTX_DOCS_TEST_DATA_DIR
// environment variable and fall back to a path relative to this source file.
static std::string get_docs_test_data_dir() {
    if (const char* env = std::getenv("OVRTX_DOCS_TEST_DATA_DIR")) {
        return env;
    }
    std::filesystem::path src(__FILE__);
    return (src.parent_path() / "../data").lexically_normal().string();
}

// Helper to make an ovx_string_t from a C string literal.
static ovx_string_t ovx_str(char const* s) {
    return {s, strlen(s)};
}

// Find a named render var output handle within a single render product's output.
static ovrtx_render_var_output_handle_t
find_product_output(ovrtx_render_product_output_t const& product_output,
                    char const* output_to_find) {
    for (size_t f = 0; f < product_output.output_frame_count; ++f) {
        ovrtx_render_product_frame_output_t const& frame =
            product_output.output_frames[f];
        for (size_t v = 0; v < frame.render_var_count; ++v) {
            ovrtx_render_product_render_var_output_t const& var =
                frame.output_render_vars[v];
            if (var.render_var_name.ptr &&
                strncmp(var.render_var_name.ptr, output_to_find,
                        var.render_var_name.length) == 0) {
                return var.output_handle;
            }
        }
    }
    return OVRTX_INVALID_HANDLE;
}

// Find a named render var output handle across all render products.
static ovrtx_render_var_output_handle_t
find_output(ovrtx_render_product_set_outputs_t const& outputs,
            char const* output_to_find) {
    for (size_t i = 0; i < outputs.output_count; ++i) {
        ovrtx_render_var_output_handle_t handle = find_product_output(outputs.outputs[i], output_to_find);
        if (handle != OVRTX_INVALID_HANDLE) {
            return handle;
        }
    }
    return OVRTX_INVALID_HANDLE;
}

// Return the _output directory. Prefer the OVRTX_TEST_OUTPUT_DIR environment
// variable (set by CI when build and test run on different machines) and fall
// back to a path relative to this source file for local development.
static std::filesystem::path get_output_dir() {
    std::filesystem::path dir;
    if (const char* env = std::getenv("OVRTX_TEST_OUTPUT_DIR")) {
        dir = env;
    } else {
        dir = std::filesystem::path(__FILE__).parent_path() / "_output";
    }
    std::filesystem::create_directories(dir);
    return dir;
}

// RAII wrapper that builds an ovrtx_config_t directing the log to
// _output/<test_name>-ovrtx.log.  Owns the backing storage so the config
// stays valid for the lifetime of the object.  An optional log_level
// (e.g. "info", "warn", "error") can be supplied to pin the threshold.
struct TestConfig {
    std::string log_path_str;
    ovx_string_t log_path;
    std::string log_level_str;
    ovx_string_t log_level;
    ovrtx_config_entry_t entries[2]{};
    ovrtx_config_t config;

    TestConfig(TestConfig const&) = delete;
    TestConfig& operator=(TestConfig const&) = delete;

    explicit TestConfig(char const* test_name, char const* log_level_value = nullptr) {
        log_path_str = (get_output_dir() / (std::string(test_name) + "-ovrtx.log")).string();
        log_path = {log_path_str.c_str(), log_path_str.size()};
        entries[0] = ovrtx_config_entry_log_file_path(log_path);
        size_t entry_count = 1;
        if (log_level_value && *log_level_value) {
            log_level_str = log_level_value;
            log_level = {log_level_str.c_str(), log_level_str.size()};
            entries[1] = ovrtx_config_entry_log_level(log_level);
            entry_count = 2;
        }
        config = {entries, entry_count};
    }
};

// Print all op errors from a wait result, then return the error count.
// Call immediately after ovrtx_wait_op() and before any assertion on
// num_error_ops so that failure messages include the actual error strings.
static std::string format_op_errors(ovrtx_op_wait_result_t const& wr) {
    std::string msg;
    for (size_t i = 0; i < wr.num_error_ops; ++i) {
        ovx_string_t err = ovrtx_get_last_op_error(wr.error_op_ids[i]);
        msg += "  op " + std::to_string(wr.error_op_ids[i]) + ": ";
        msg += std::string(err.ptr, err.length) + "\n";
    }
    return msg;
}

// Format the last per-thread API error string. Use after any enqueue- or
// synchronous-call failure (status != OVRTX_API_SUCCESS) to surface the
// underlying reason. The string is valid until the next API call on this
// thread, so capture it immediately.
static std::string format_api_error() {
    ovx_string_t err = ovrtx_get_last_error();
    if (err.ptr && err.length > 0) {
        return "  ovrtx_get_last_error: " + std::string(err.ptr, err.length) + "\n";
    }
    return "  (ovrtx_get_last_error returned empty)\n";
}

static char const* api_status_name(ovrtx_api_status_t status) {
    switch (status) {
        case OVRTX_API_SUCCESS:
            return "OVRTX_API_SUCCESS";
        case OVRTX_API_ERROR:
            return "OVRTX_API_ERROR";
        case OVRTX_API_TIMEOUT:
            return "OVRTX_API_TIMEOUT";
    }
    return "OVRTX_API_UNKNOWN";
}

static std::string format_api_status_error(ovrtx_api_status_t status) {
    std::string msg = "  ovrtx status: ";
    msg += api_status_name(status);
    msg += " (";
    msg += std::to_string(static_cast<int>(status));
    msg += ")\n";
    msg += format_api_error();
    return msg;
}

// Convenience macro: assert no op errors, printing the error strings on failure.
#define ASSERT_NO_OP_ERRORS(wr) \
    do {                                                                     \
        ovrtx_op_wait_result_t const& wr__ = (wr);                           \
        std::string const op_errors__ = format_op_errors(wr__);              \
        ASSERT_EQ(wr__.num_error_ops, 0u) << op_errors__;                    \
    } while (false)

// Convenience macros: assert/expect that an API call returned OVRTX_API_SUCCESS,
// printing ovrtx_get_last_error() on failure so the runtime reason is visible.
// Works for any call site whose status field is `ovrtx_api_status_t`
// (ovrtx_enqueue_result_t::status, ovrtx_result_t::status, etc.).
#define ASSERT_API_SUCCESS(status_expr)                                      \
    do {                                                                     \
        ovrtx_api_status_t const status__ = (status_expr);                   \
        std::string const api_error__ =                                      \
            status__ == OVRTX_API_SUCCESS ? std::string() : format_api_status_error(status__); \
        ASSERT_EQ(status__, OVRTX_API_SUCCESS) << api_error__;               \
    } while (false)

#define EXPECT_API_SUCCESS(status_expr)                                      \
    do {                                                                     \
        ovrtx_api_status_t const status__ = (status_expr);                   \
        std::string const api_error__ =                                      \
            status__ == OVRTX_API_SUCCESS ? std::string() : format_api_status_error(status__); \
        EXPECT_EQ(status__, OVRTX_API_SUCCESS) << api_error__;               \
    } while (false)

// Save an RGBA uint8 LdrColor buffer to _output/<name>.png.
static void save_ldr_png(char const* name, void const* data, int width, int height) {
    auto path = get_output_dir() / (std::string(name) + ".png");
    int ok = stbi_write_png(path.string().c_str(), width, height, 4, data, width * 4);
    if (ok) {
        printf("Saved %s\n", path.string().c_str());
    } else {
        printf("WARNING: failed to write %s\n", path.string().c_str());
    }
}

// Build an inline USDA string that sublayers a scene file and adds a
// RenderProduct with the given render vars.
static std::string make_sublayer_usda(
    std::string_view scene_path,
    std::string_view render_product_body) {
    std::string usda;
    usda += "#usda 1.0\n";
    usda += "(\n";
    usda += "    subLayers = [\n";
    usda += "        @";
    usda += scene_path;
    usda += "@\n";
    usda += "    ]\n";
    usda += ")\n\n";
    usda += render_product_body;
    usda += "\n";
    return usda;
}

// Resolve a single ovx_primpath_t into a "/A/B/C" string via the path dictionary.
// [snippet:doc-resolve-primpath-helper-c]
static std::string docs_resolve_primpath(path_dictionary_instance_t* pd, ovx_primpath_t p) {
    ovx_token_t token_buf[64];
    ovx_token_t* tokens_out = nullptr;
    size_t num_tokens = 0;
    size_t num_processed = 0;
    ovx_api_result_t r = path_dictionary_get_tokens_from_paths(
        pd, &p, 1, token_buf, 64, &tokens_out, &num_tokens, &num_processed);
    if (r.status != OVX_API_SUCCESS || num_processed == 0) {
        return "";
    }
    std::string out;
    for (size_t i = 0; i < num_tokens; ++i) {
        ovx_string_t s{};
        if (path_dictionary_get_strings_from_tokens(pd, &tokens_out[i], 1, &s).status ==
            OVX_API_SUCCESS) {
            out += "/";
            out.append(s.ptr, s.length);
        }
    }
    return out;
}
// [/snippet:doc-resolve-primpath-helper-c]

// Collect every resolved prim path from an ovrtx query result. The query result
// must still be alive while this runs because the group prim-list handles are
// owned by the query result.
static std::set<std::string> docs_collect_query_paths(ovrtx_query_result_t const& qr,
                                                      path_dictionary_instance_t* pd) {
    std::set<std::string> paths;
    for (size_t g = 0; g < qr.group_count; ++g) {
        ovrtx_query_prim_group_t const& group = qr.groups[g];
        size_t num_paths = 0;
        if (path_dictionary_get_num_paths_from_path_list(pd, group.prim_list_handle, &num_paths)
                .status != OVX_API_SUCCESS) {
            continue;
        }
        std::vector<ovx_primpath_t> prim_paths(num_paths);
        size_t out_num = 0;
        if (path_dictionary_get_paths_from_path_list(pd, group.prim_list_handle, 0, num_paths,
                                                     prim_paths.data(), &out_num)
                .status != OVX_API_SUCCESS) {
            continue;
        }
        for (size_t i = 0; i < out_num; ++i) {
            paths.insert(docs_resolve_primpath(pd, prim_paths[i]));
        }
    }
    return paths;
}

// Query all prim paths currently visible to the runtime stage.
static std::set<std::string> docs_query_all_paths(ovrtx_renderer_t* renderer) {
    ovrtx_query_desc_t desc{};
    desc.attribute_filter.mode = OVRTX_ATTRIBUTE_FILTER_NONE;

    ovrtx_query_handle_t query_handle = 0;
    ovrtx_enqueue_result_t eq = ovrtx_query_prims(renderer, &desc, &query_handle);
    if (eq.status != OVRTX_API_SUCCESS) {
        return {};
    }

    ovrtx_op_wait_result_t wait_result{};
    if (ovrtx_wait_op(renderer, eq.op_index, ovrtx_timeout_infinite, &wait_result).status !=
        OVRTX_API_SUCCESS) {
        return {};
    }
    if (wait_result.num_error_ops != 0) {
        return {};
    }

    ovrtx_query_result_t qr{};
    if (ovrtx_fetch_query_results(renderer, query_handle, ovrtx_timeout_infinite, &qr).status !=
        OVRTX_API_SUCCESS) {
        return {};
    }

    path_dictionary_instance_t pd{};
    std::set<std::string> paths;
    if (ovrtx_get_path_dictionary(renderer, &pd).status == OVRTX_API_SUCCESS) {
        paths = docs_collect_query_paths(qr, &pd);
    }
    ovrtx_release_query_results(renderer, query_handle);
    return paths;
}

static void docs_wait_no_errors(ovrtx_renderer_t* renderer, ovrtx_op_id_t op_id) {
    ovrtx_op_wait_result_t wait_result{};
    ASSERT_API_SUCCESS(ovrtx_wait_op(renderer, op_id, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);
}
