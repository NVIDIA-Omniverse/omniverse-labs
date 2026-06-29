// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Tests for persistent attribute bindings and attribute mapping through the C API.

#include <gtest/gtest.h>
#include "helpers.h"

#include <ovrtx/ovrtx_attributes.h>

#include <cstring>
#include <string>

namespace {

void read_xform_translation_x(ovrtx_renderer_t* renderer, char const* prim_path, double* out_x) {
    ovx_string_t prim = ovx_str(prim_path);
    DLDataType mat_type = {kDLFloat, 64, 16};
    ovrtx_binding_desc_or_handle_t binding =
        ovrtx_make_binding_desc(&prim, 1, ovx_str("omni:xform"), OVRTX_SEMANTIC_NONE, mat_type);

    ovrtx_read_handle_t read_handle = 0;
    ovrtx_enqueue_result_t eq = ovrtx_read_attribute(renderer, &binding, nullptr, &read_handle);
    ASSERT_API_SUCCESS(eq.status);
    docs_wait_no_errors(renderer, eq.op_index);

    ovrtx_read_output_t output{};
    ASSERT_API_SUCCESS(
        ovrtx_fetch_read_result(renderer, read_handle, ovrtx_timeout_infinite, &output).status);
    double const* matrix = static_cast<double const*>(output.buffers[0].dl.data);
    *out_x = matrix[12];

    ovrtx_cuda_sync_t no_sync{};
    ASSERT_API_SUCCESS(ovrtx_release_read_result(renderer, output.map_handle, no_sync).status);
}

void make_xform(double x, double out[16]) {
    std::memset(out, 0, sizeof(double) * 16);
    out[0] = 1.0;
    out[5] = 1.0;
    out[10] = 1.0;
    out[15] = 1.0;
    out[12] = x;
}

} // namespace

class AttributeBindingsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        TestConfig tc("AttributeBindingsTest");
        ovrtx_result_t result = ovrtx_create_renderer(&tc.config, &renderer_);
        ASSERT_API_SUCCESS(result.status);
    }

    static void TearDownTestSuite() {
        if (renderer_) {
            ovrtx_destroy_renderer(renderer_);
            renderer_ = nullptr;
        }
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

ovrtx_renderer_t* AttributeBindingsTest::renderer_ = nullptr;

TEST_F(AttributeBindingsTest, CreateWriteDestroyBinding) {
    load_base();

    ovx_string_t prim = ovx_str("/World/Plane");
    DLDataType mat_type = {kDLFloat, 64, 16};
    ovrtx_binding_desc_or_handle_t binding_desc =
        ovrtx_make_binding_desc(&prim, 1, ovx_str("omni:xform"), OVRTX_SEMANTIC_XFORM_MAT4x4, mat_type);
    binding_desc.binding_desc.flags = OVRTX_BINDING_FLAG_OPTIMIZE;

    // [snippet:doc-create-attribute-binding-c]
    ovrtx_attribute_binding_handle_t binding_handle = 0;
    ovrtx_enqueue_result_t eq =
        ovrtx_create_attribute_binding(renderer_, &binding_desc.binding_desc, &binding_handle);
    ASSERT_API_SUCCESS(eq.status);
    docs_wait_no_errors(renderer_, eq.op_index);
    // [/snippet:doc-create-attribute-binding-c]

    // [snippet:doc-write-bound-attribute-c]
    ovrtx_binding_desc_or_handle_t binding_ref{};
    binding_ref.binding_handle = binding_handle;

    double matrix[16];
    make_xform(14.0, matrix);
    size_t count = 1;
    DLTensor tensor = ovrtx_make_write_cpu_tensor(matrix, &count, mat_type);
    ovrtx_input_buffer_t buffer{};
    buffer.tensors = &tensor;
    buffer.tensor_count = 1;

    eq = ovrtx_write_attribute(renderer_, &binding_ref, &buffer, OVRTX_DATA_ACCESS_SYNC);
    ASSERT_API_SUCCESS(eq.status);
    docs_wait_no_errors(renderer_, eq.op_index);
    // [/snippet:doc-write-bound-attribute-c]

    double x = 0.0;
    read_xform_translation_x(renderer_, "/World/Plane", &x);
    EXPECT_DOUBLE_EQ(x, 14.0);

    // [snippet:doc-destroy-attribute-binding-c]
    eq = ovrtx_destroy_attribute_binding(renderer_, binding_handle);
    ASSERT_API_SUCCESS(eq.status);
    docs_wait_no_errors(renderer_, eq.op_index);
    // [/snippet:doc-destroy-attribute-binding-c]
}

TEST_F(AttributeBindingsTest, MapAndUnmapAttribute) {
    load_base();

    ovx_string_t prim = ovx_str("/World/Plane");
    DLDataType mat_type = {kDLFloat, 64, 16};
    ovrtx_binding_desc_or_handle_t binding =
        ovrtx_make_binding_desc(&prim, 1, ovx_str("omni:xform"), OVRTX_SEMANTIC_XFORM_MAT4x4, mat_type);

    // [snippet:doc-map-attribute-cpu-c]
    ovrtx_mapping_desc_t mapping_desc{};
    mapping_desc.device_type = kDLCPU;
    mapping_desc.device_id = 0;

    ovrtx_attribute_mapping_t mapping{};
    ovrtx_result_t result = ovrtx_map_attribute(renderer_, &binding, mapping_desc, &mapping);
    ASSERT_API_SUCCESS(result.status);

    double* matrix = static_cast<double*>(mapping.dl.data);
    matrix[0] = 1.0;
    matrix[5] = 1.0;
    matrix[10] = 1.0;
    matrix[15] = 1.0;
    matrix[12] = 15.0;

    ovrtx_cuda_sync_t no_sync{};
    ovrtx_enqueue_result_t eq = ovrtx_unmap_attribute(renderer_, mapping.map_handle, no_sync);
    ASSERT_API_SUCCESS(eq.status);
    docs_wait_no_errors(renderer_, eq.op_index);
    // [/snippet:doc-map-attribute-cpu-c]

    double x = 0.0;
    read_xform_translation_x(renderer_, "/World/Plane", &x);
    EXPECT_DOUBLE_EQ(x, 15.0);
}
