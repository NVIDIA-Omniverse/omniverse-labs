// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Tests for additive USD references, removal, and cloning through the C API.

#include <gtest/gtest.h>
#include "helpers.h"

#include <ovrtx/ovrtx_attributes.h>

#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr char kRootUsda[] = R"(#usda 1.0
def Xform "World" {
}
)";

constexpr char kReferenceUsda[] = R"(#usda 1.0
(
    defaultPrim = "Referenced"
)

def Xform "Referenced" {
    def Cube "KnownChild" {
    }
}
)";

void read_plane_points(ovrtx_renderer_t* renderer, char const* prim_path, std::vector<float>* values) {
    ovx_string_t prim = ovx_str(prim_path);
    DLDataType point3f_type = {kDLFloat, 32, 3};
    ovrtx_binding_desc_or_handle_t binding = ovrtx_make_binding_desc(
        &prim, 1, ovx_str("points"), OVRTX_SEMANTIC_NONE, point3f_type);
    binding.binding_desc.attribute_type.is_array = true;

    ovrtx_read_handle_t read_handle = 0;
    ovrtx_enqueue_result_t eq = ovrtx_read_attribute(renderer, &binding, nullptr, &read_handle);
    ASSERT_API_SUCCESS(eq.status);
    docs_wait_no_errors(renderer, eq.op_index);

    ovrtx_read_output_t output{};
    ASSERT_API_SUCCESS(
        ovrtx_fetch_read_result(renderer, read_handle, ovrtx_timeout_infinite, &output).status);
    ASSERT_EQ(output.buffer_count, 1u);

    DLTensor const& t = output.buffers[0].dl;
    int64_t scalar_count = t.dtype.lanes;
    for (int d = 0; d < t.ndim; ++d) {
        scalar_count *= t.shape[d];
    }
    float const* data = static_cast<float const*>(t.data);
    values->assign(data, data + scalar_count);

    ovrtx_cuda_sync_t no_sync{};
    ASSERT_API_SUCCESS(ovrtx_release_read_result(renderer, output.map_handle, no_sync).status);
}

} // namespace

class StageMutationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        TestConfig tc("StageMutationTest");
        ovrtx_result_t result = ovrtx_create_renderer(&tc.config, &renderer_);
        ASSERT_API_SUCCESS(result.status);
    }

    static void TearDownTestSuite() {
        if (renderer_) {
            ovrtx_destroy_renderer(renderer_);
            renderer_ = nullptr;
        }
    }

    static void open_root() {
        ovrtx_enqueue_result_t eq =
            ovrtx_open_usd_from_string(renderer_, {kRootUsda, strlen(kRootUsda)});
        ASSERT_API_SUCCESS(eq.status);
        docs_wait_no_errors(renderer_, eq.op_index);
        eq = ovrtx_reset(renderer_, 0.0);
        ASSERT_API_SUCCESS(eq.status);
        docs_wait_no_errors(renderer_, eq.op_index);
    }

    static void load_base() {
        ovrtx_enqueue_result_t eq = ovrtx_reset_stage(renderer_);
        ASSERT_API_SUCCESS(eq.status);
        docs_wait_no_errors(renderer_, eq.op_index);

        std::string scene = get_docs_test_data_dir() + "/ovrtx-test-base.usda";
        eq = ovrtx_open_usd_from_file(renderer_, {scene.c_str(), scene.size()});
        ASSERT_API_SUCCESS(eq.status);
        docs_wait_no_errors(renderer_, eq.op_index);

        eq = ovrtx_reset(renderer_, 0.0);
        ASSERT_API_SUCCESS(eq.status);
        docs_wait_no_errors(renderer_, eq.op_index);
    }

    static ovrtx_renderer_t* renderer_;
};

ovrtx_renderer_t* StageMutationTest::renderer_ = nullptr;

TEST_F(StageMutationTest, AddRemoveUsdReferenceFromFile) {
    open_root();
    std::filesystem::path reference_path = get_output_dir() / "stage-mutation-reference.usda";
    std::ofstream(reference_path) << kReferenceUsda;
    std::string reference_path_str = reference_path.string();

    // [snippet:doc-add-remove-usd-reference-c]
    ovrtx_usd_handle_t usd_handle{};
    ovrtx_enqueue_result_t eq = ovrtx_add_usd_reference_from_file(
        renderer_,
        {reference_path_str.c_str(), reference_path_str.size()},
        ovx_str("/World/LoadedBase"),
        &usd_handle);
    ASSERT_API_SUCCESS(eq.status);
    docs_wait_no_errors(renderer_, eq.op_index);

    std::set<std::string> paths = docs_query_all_paths(renderer_);
    ASSERT_TRUE(paths.count("/World/LoadedBase"));
    ASSERT_TRUE(paths.count("/World/LoadedBase/KnownChild"));

    eq = ovrtx_remove_usd(renderer_, usd_handle);
    ASSERT_API_SUCCESS(eq.status);
    docs_wait_no_errors(renderer_, eq.op_index);
    // [/snippet:doc-add-remove-usd-reference-c]

    paths = docs_query_all_paths(renderer_);
    EXPECT_FALSE(paths.count("/World/LoadedBase"));
    EXPECT_FALSE(paths.count("/World/LoadedBase/KnownChild"));
}

TEST_F(StageMutationTest, AddRemoveUsdReferenceFromString) {
    open_root();

    // [snippet:doc-add-usd-reference-from-string-c]
    ovrtx_usd_handle_t usd_handle{};
    ovrtx_enqueue_result_t eq = ovrtx_add_usd_reference_from_string(
        renderer_,
        {kReferenceUsda, strlen(kReferenceUsda)},
        ovx_str("/World/Injected"),
        &usd_handle);
    ASSERT_API_SUCCESS(eq.status);
    docs_wait_no_errors(renderer_, eq.op_index);

    std::set<std::string> paths = docs_query_all_paths(renderer_);
    ASSERT_TRUE(paths.count("/World/Injected"));
    ASSERT_TRUE(paths.count("/World/Injected/KnownChild"));

    eq = ovrtx_remove_usd(renderer_, usd_handle);
    ASSERT_API_SUCCESS(eq.status);
    docs_wait_no_errors(renderer_, eq.op_index);
    // [/snippet:doc-add-usd-reference-from-string-c]

    paths = docs_query_all_paths(renderer_);
    EXPECT_FALSE(paths.count("/World/Injected"));
    EXPECT_FALSE(paths.count("/World/Injected/KnownChild"));
}

TEST_F(StageMutationTest, CloneUsd) {
    load_base();
    std::vector<float> source_points;
    read_plane_points(renderer_, "/World/Plane", &source_points);

    // [snippet:doc-clone-usd-c]
    ovx_string_t source = ovx_str("/World/Plane");
    ovx_string_t targets[] = {
        ovx_str("/World/PlaneCloneA"),
        ovx_str("/World/PlaneCloneB"),
    };

    ovrtx_enqueue_result_t eq = ovrtx_clone_usd(renderer_, source, targets, 2);
    ASSERT_API_SUCCESS(eq.status);
    docs_wait_no_errors(renderer_, eq.op_index);
    // [/snippet:doc-clone-usd-c]

    std::set<std::string> paths = docs_query_all_paths(renderer_);
    EXPECT_TRUE(paths.count("/World/PlaneCloneA"));
    EXPECT_TRUE(paths.count("/World/PlaneCloneB"));

    std::vector<float> clone_points;
    read_plane_points(renderer_, "/World/PlaneCloneA", &clone_points);
    EXPECT_EQ(clone_points, source_points);
}
