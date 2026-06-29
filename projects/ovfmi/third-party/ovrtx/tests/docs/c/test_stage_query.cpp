// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Tests for ovrtx_query_prims / ovrtx_fetch_query_results / ovrtx_release_query_results
// and the path dictionary accessor ovrtx_get_path_dictionary.

#include <gtest/gtest.h>
#include "helpers.h"

#include <ovrtx/ovrtx_attributes.h>
#include <ovx/path_dictionary/path_dictionary_utils.h>

#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

// Resolve a single ovx_primpath_t into a "/A/B/C" string via the path dictionary.
std::string resolve_primpath(path_dictionary_instance_t* pd, ovx_primpath_t p) {
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

std::set<std::string> collect_paths(ovrtx_query_result_t const& qr,
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
            paths.insert(resolve_primpath(pd, prim_paths[i]));
        }
    }
    return paths;
}

} // anonymous namespace

class StageQueryTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        TestConfig tc("StageQueryTest");
        ovrtx_result_t result = ovrtx_create_renderer(&tc.config, &renderer_);
        ASSERT_API_SUCCESS(result.status);
    }

    static void TearDownTestSuite() {
        if (renderer_) {
            ovrtx_destroy_renderer(renderer_);
            renderer_ = nullptr;
        }
    }

    // Reset the stage and reload ovrtx-test-base.usda.
    static void load_base() {
        ovrtx_op_wait_result_t wait_result;
        ovrtx_enqueue_result_t eq;

        eq = ovrtx_reset_stage(renderer_);
        ASSERT_API_SUCCESS(eq.status);
        ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
        ASSERT_NO_OP_ERRORS(wait_result);

        std::string scene = get_docs_test_data_dir() + "/ovrtx-test-base.usda";
        eq = ovrtx_open_usd_from_file(renderer_, {scene.c_str(), scene.size()});
        ASSERT_API_SUCCESS(eq.status);
        ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
        ASSERT_NO_OP_ERRORS(wait_result);

        eq = ovrtx_reset(renderer_, 0.0);
        ASSERT_API_SUCCESS(eq.status);
        ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
        ASSERT_NO_OP_ERRORS(wait_result);
    }

    static ovrtx_renderer_t* renderer_;
};

ovrtx_renderer_t* StageQueryTest::renderer_ = nullptr;

TEST_F(StageQueryTest, QueryAllPrimsBasic) {
    load_base();

    // [snippet:doc-query-prims-basic-c]
    // Issue a query with no filters and no attribute reporting.
    ovrtx_query_desc_t desc{};
    desc.attribute_filter.mode = OVRTX_ATTRIBUTE_FILTER_NONE;

    ovrtx_query_handle_t query_handle = 0;
    ovrtx_enqueue_result_t eq = ovrtx_query_prims(renderer_, &desc, &query_handle);
    ASSERT_API_SUCCESS(eq.status);

    ovrtx_op_wait_result_t wait_result;
    ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    ovrtx_query_result_t qr{};
    ASSERT_API_SUCCESS(ovrtx_fetch_query_results(renderer_, query_handle, ovrtx_timeout_infinite, &qr).status);

    // Each group carries a prim_list_handle that can be plugged into
    // ovrtx_binding_desc_t::prims_list_handle for subsequent reads/writes.
    for (size_t g = 0; g < qr.group_count; ++g) {
        printf("group %zu: %zu prims\n", g, qr.groups[g].prim_count);
    }

    // Always release the query when done — the prim list handles are owned
    // by this result and become invalid after release.
    ASSERT_API_SUCCESS(ovrtx_release_query_results(renderer_, query_handle).status);
    // [/snippet:doc-query-prims-basic-c]

    EXPECT_GT(qr.total_prim_count, 0u);
}

TEST_F(StageQueryTest, QueryByPrimType) {
    load_base();

    // [snippet:doc-query-prims-by-type-c]
    // AND filter: require prim type == "Mesh".
    ovx_string_t mesh_type = ovx_str("Mesh");
    ovrtx_filter_t filter{};
    filter.kind = OVRTX_FILTER_PRIM_TYPE;
    filter.name.string = mesh_type;

    ovrtx_query_desc_t desc{};
    desc.require_all = &filter;
    desc.require_all_count = 1;
    desc.attribute_filter.mode = OVRTX_ATTRIBUTE_FILTER_NONE;

    ovrtx_query_handle_t query_handle = 0;
    ovrtx_enqueue_result_t eq = ovrtx_query_prims(renderer_, &desc, &query_handle);
    ASSERT_API_SUCCESS(eq.status);

    ovrtx_op_wait_result_t wait_result;
    ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    ovrtx_query_result_t qr{};
    ASSERT_API_SUCCESS(ovrtx_fetch_query_results(renderer_, query_handle, ovrtx_timeout_infinite, &qr).status);
    // [/snippet:doc-query-prims-by-type-c]

    EXPECT_GT(qr.total_prim_count, 0u) << "expected at least one Mesh prim";

    path_dictionary_instance_t pd{};
    ASSERT_API_SUCCESS(ovrtx_get_path_dictionary(renderer_, &pd).status);
    std::set<std::string> paths = collect_paths(qr, &pd);
    EXPECT_TRUE(paths.count("/World/Plane") == 1u) << "expected /World/Plane in mesh query";

    ASSERT_API_SUCCESS(ovrtx_release_query_results(renderer_, query_handle).status);
}

TEST_F(StageQueryTest, QueryHasAttribute) {
    load_base();

    // [snippet:doc-query-has-attribute-c]
    // Match prims that expose a "points" attribute.
    ovx_string_t points_attr = ovx_str("points");
    ovrtx_filter_t filter{};
    filter.kind = OVRTX_FILTER_HAS_ATTRIBUTE;
    filter.name.string = points_attr;

    ovrtx_query_desc_t desc{};
    desc.require_all = &filter;
    desc.require_all_count = 1;
    desc.attribute_filter.mode = OVRTX_ATTRIBUTE_FILTER_NONE;
    // [/snippet:doc-query-has-attribute-c]

    ovrtx_query_handle_t query_handle = 0;
    ovrtx_enqueue_result_t eq = ovrtx_query_prims(renderer_, &desc, &query_handle);
    ASSERT_API_SUCCESS(eq.status);

    ovrtx_op_wait_result_t wait_result;
    ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    ovrtx_query_result_t qr{};
    ASSERT_API_SUCCESS(ovrtx_fetch_query_results(renderer_, query_handle, ovrtx_timeout_infinite, &qr).status);

    path_dictionary_instance_t pd{};
    ASSERT_API_SUCCESS(ovrtx_get_path_dictionary(renderer_, &pd).status);
    std::set<std::string> paths = collect_paths(qr, &pd);
    EXPECT_TRUE(paths.count("/World/Plane") == 1u);

    ASSERT_API_SUCCESS(ovrtx_release_query_results(renderer_, query_handle).status);
}

TEST_F(StageQueryTest, QueryRequireAnyExcludeAllAttrs) {
    load_base();

    // [snippet:doc-query-require-any-exclude-c]
    // Match Mesh or Camera prims, then exclude Camera. The exclusion removes
    // a prim that would otherwise match the OR clause.
    ovx_string_t mesh_type = ovx_str("Mesh");
    ovx_string_t camera_type = ovx_str("Camera");
    ovrtx_filter_t any_filters[2]{};
    any_filters[0].kind = OVRTX_FILTER_PRIM_TYPE;
    any_filters[0].name.string = mesh_type;
    any_filters[1].kind = OVRTX_FILTER_PRIM_TYPE;
    any_filters[1].name.string = camera_type;

    ovrtx_filter_t exclude_filter{};
    exclude_filter.kind = OVRTX_FILTER_PRIM_TYPE;
    exclude_filter.name.string = camera_type;

    ovrtx_query_desc_t desc{};
    desc.require_any = any_filters;
    desc.require_any_count = 2;
    desc.exclude = &exclude_filter;
    desc.exclude_count = 1;
    desc.attribute_filter.mode = OVRTX_ATTRIBUTE_FILTER_ALL;
    // [/snippet:doc-query-require-any-exclude-c]

    ovrtx_query_handle_t query_handle = 0;
    ovrtx_enqueue_result_t eq = ovrtx_query_prims(renderer_, &desc, &query_handle);
    ASSERT_API_SUCCESS(eq.status);

    ovrtx_op_wait_result_t wait_result;
    ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    ovrtx_query_result_t qr{};
    ASSERT_API_SUCCESS(ovrtx_fetch_query_results(renderer_, query_handle, ovrtx_timeout_infinite, &qr).status);

    path_dictionary_instance_t pd{};
    ASSERT_API_SUCCESS(ovrtx_get_path_dictionary(renderer_, &pd).status);
    std::set<std::string> paths = collect_paths(qr, &pd);
    EXPECT_TRUE(paths.count("/World/Plane") == 1u);
    EXPECT_FALSE(paths.count("/World/Camera") == 1u);
    ASSERT_GT(qr.group_count, 0u);
    EXPECT_GT(qr.groups[0].attribute_count, 0u);

    ASSERT_API_SUCCESS(ovrtx_release_query_results(renderer_, query_handle).status);
}

TEST_F(StageQueryTest, QuerySpecificEmptyAttributes) {
    load_base();

    // [snippet:doc-query-specific-empty-attributes-c]
    // SPECIFIC with an empty attribute list returns matching prims without
    // any attribute descriptors.
    ovx_string_t mesh_type = ovx_str("Mesh");
    ovrtx_filter_t filter{};
    filter.kind = OVRTX_FILTER_PRIM_TYPE;
    filter.name.string = mesh_type;

    ovrtx_query_desc_t desc{};
    desc.require_all = &filter;
    desc.require_all_count = 1;
    desc.attribute_filter.mode = OVRTX_ATTRIBUTE_FILTER_SPECIFIC;
    desc.attribute_filter.attribute_names = nullptr;
    desc.attribute_filter.attribute_name_count = 0;
    // [/snippet:doc-query-specific-empty-attributes-c]

    ovrtx_query_handle_t query_handle = 0;
    ovrtx_enqueue_result_t eq = ovrtx_query_prims(renderer_, &desc, &query_handle);
    ASSERT_API_SUCCESS(eq.status);

    ovrtx_op_wait_result_t wait_result;
    ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    ovrtx_query_result_t qr{};
    ASSERT_API_SUCCESS(ovrtx_fetch_query_results(renderer_, query_handle, ovrtx_timeout_infinite, &qr).status);

    path_dictionary_instance_t pd{};
    ASSERT_API_SUCCESS(ovrtx_get_path_dictionary(renderer_, &pd).status);
    std::set<std::string> paths = collect_paths(qr, &pd);
    EXPECT_TRUE(paths.count("/World/Plane") == 1u);
    for (size_t g = 0; g < qr.group_count; ++g) {
        EXPECT_EQ(qr.groups[g].attribute_count, 0u);
    }

    ASSERT_API_SUCCESS(ovrtx_release_query_results(renderer_, query_handle).status);
}

TEST_F(StageQueryTest, PathDictionaryResolve) {
    load_base();

    // Query the scene's Camera so we have a small, known prim_list_handle to resolve.
    ovx_string_t camera_type = ovx_str("Camera");
    ovrtx_filter_t filter{};
    filter.kind = OVRTX_FILTER_PRIM_TYPE;
    filter.name.string = camera_type;

    ovrtx_query_desc_t desc{};
    desc.require_all = &filter;
    desc.require_all_count = 1;
    desc.attribute_filter.mode = OVRTX_ATTRIBUTE_FILTER_NONE;

    ovrtx_query_handle_t query_handle = 0;
    ovrtx_enqueue_result_t eq = ovrtx_query_prims(renderer_, &desc, &query_handle);
    ASSERT_API_SUCCESS(eq.status);

    ovrtx_op_wait_result_t wait_result;
    ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    ovrtx_query_result_t qr{};
    ASSERT_API_SUCCESS(ovrtx_fetch_query_results(renderer_, query_handle, ovrtx_timeout_infinite, &qr).status);
    ASSERT_GE(qr.group_count, 1u);

    // [snippet:doc-path-dictionary-resolve-c]
    // The renderer's path dictionary converts between string paths and internal
    // handles. It is valid for the lifetime of the renderer — no release is required.
    path_dictionary_instance_t pd{};
    ASSERT_API_SUCCESS(ovrtx_get_path_dictionary(renderer_, &pd).status);

    // 1) Resolve the primpath handles in a query result's prim_list_handle to
    //    "/A/B/C" string paths.
    ovx_primpath_list_t handle = qr.groups[0].prim_list_handle;
    size_t num_paths = 0;
    ASSERT_EQ(path_dictionary_get_num_paths_from_path_list(&pd, handle, &num_paths).status,
              OVX_API_SUCCESS);

    std::vector<ovx_primpath_t> prim_paths(num_paths);
    size_t out_num = 0;
    ASSERT_EQ(path_dictionary_get_paths_from_path_list(&pd, handle, 0, num_paths,
                                                       prim_paths.data(), &out_num)
                  .status,
              OVX_API_SUCCESS);

    // A primpath decomposes into a sequence of tokens; join them with '/' to get
    // a full stage path.
    std::vector<std::string> path_strings;
    for (size_t i = 0; i < out_num; ++i) {
        ovx_token_t token_buf[64];
        ovx_token_t* tokens_out = nullptr;
        size_t num_tokens = 0;
        size_t num_processed = 0;
        ASSERT_EQ(path_dictionary_get_tokens_from_paths(&pd, &prim_paths[i], 1, token_buf, 64,
                                                        &tokens_out, &num_tokens, &num_processed)
                      .status,
                  OVX_API_SUCCESS);
        std::string s;
        for (size_t t = 0; t < num_tokens; ++t) {
            ovx_string_t tok_s{};
            ASSERT_EQ(path_dictionary_get_strings_from_tokens(&pd, &tokens_out[t], 1, &tok_s)
                          .status,
                      OVX_API_SUCCESS);
            s += "/";
            s.append(tok_s.ptr, tok_s.length);
        }
        path_strings.push_back(s);
    }

    // 2) Round-trip: rebuild a path list from the resolved strings and verify
    //    it holds the same number of paths.
    std::vector<ovx_string_t> str_views(path_strings.size());
    for (size_t i = 0; i < path_strings.size(); ++i) {
        str_views[i] = {path_strings[i].c_str(), path_strings[i].size()};
    }
    ovx_primpath_list_t rebuilt{};
    ASSERT_EQ(path_dictionary_create_path_list_from_strings(&pd, str_views.data(),
                                                            str_views.size(), &rebuilt)
                  .status,
              OVX_API_SUCCESS);

    size_t rebuilt_num = 0;
    ASSERT_EQ(path_dictionary_get_num_paths_from_path_list(&pd, rebuilt, &rebuilt_num).status,
              OVX_API_SUCCESS);
    EXPECT_EQ(rebuilt_num, out_num);

    ASSERT_EQ(path_dictionary_destroy_path_list(&pd, rebuilt).status, OVX_API_SUCCESS);
    // [/snippet:doc-path-dictionary-resolve-c]

    // The single Camera prim in the base scene is /World/Camera.
    ASSERT_EQ(path_strings.size(), 1u);
    EXPECT_EQ(path_strings[0], "/World/Camera");

    ASSERT_API_SUCCESS(ovrtx_release_query_results(renderer_, query_handle).status);
}
