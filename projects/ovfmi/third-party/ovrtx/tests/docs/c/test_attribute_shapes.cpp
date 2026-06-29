// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Round-trip tests for the C API attribute tensor-layout convention.
//
// The C API uses DLTensor lanes for multi-component attribute data: an
// N-element array of float3 values is shape=[N] with dtype.lanes=3, and an
// N-prim array of 4x4 matrices is shape=[N] with dtype.lanes=16. Rendered
// outputs/AOVs are the exception: image tensors use channel-last shapes such
// as [height, width, channels] with dtype.lanes=1.

#include <gtest/gtest.h>
#include "helpers.h"

#include <ovrtx/ovrtx_attributes.h>

#include <cstdint>
#include <string>

class AttributeShapesTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        TestConfig tc("AttributeShapesTest");
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

ovrtx_renderer_t* AttributeShapesTest::renderer_ = nullptr;

TEST_F(AttributeShapesTest, ScalarInt32) {
    load_base();

    // [snippet:doc-shape-scalar-int32-c]
    // Scalar per-prim attribute: ndim=1, shape=[N], dtype.lanes=1.
    ovx_string_t rp = ovx_str("/Render/Camera");
    uint32_t write_value = 23;
    int64_t shape[1] = {1};
    int64_t strides[1] = {1};

    DLTensor write_tensor{};
    write_tensor.data = &write_value;
    write_tensor.device = {kDLCPU, 0};
    write_tensor.ndim = 1;
    write_tensor.dtype = {kDLUInt, 32, 1};
    write_tensor.shape = shape;
    write_tensor.strides = strides;

    ovrtx_input_buffer_t buffer{};
    buffer.tensors = &write_tensor;
    buffer.tensor_count = 1;

    ovrtx_binding_desc_or_handle_t binding = ovrtx_make_binding_desc(
        &rp, 1, ovx_str("omni:rtx:rtpt:maxBounces"),
        OVRTX_SEMANTIC_NONE, write_tensor.dtype);

    ovrtx_enqueue_result_t eq =
        ovrtx_write_attribute(renderer_, &binding, &buffer, OVRTX_DATA_ACCESS_SYNC);
    ASSERT_API_SUCCESS(eq.status);
    ovrtx_op_wait_result_t wait_result;
    ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    // Read back — expect ndim=1, shape=[1], dtype.lanes=1.
    ovrtx_read_handle_t read_handle = 0;
    eq = ovrtx_read_attribute(renderer_, &binding, nullptr, &read_handle);
    ASSERT_API_SUCCESS(eq.status);
    ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    ovrtx_read_output_t output{};
    ASSERT_API_SUCCESS(ovrtx_fetch_read_result(renderer_, read_handle, ovrtx_timeout_infinite, &output).status);

    DLTensor const& t = output.buffers[0].dl;
    EXPECT_EQ(t.ndim, 1);
    EXPECT_EQ(t.shape[0], 1);
    EXPECT_EQ(t.dtype.lanes, 1u);
    EXPECT_EQ(*static_cast<uint32_t const*>(t.data), 23u);

    ovrtx_cuda_sync_t no_sync{};
    ASSERT_API_SUCCESS(ovrtx_release_read_result(renderer_, output.map_handle, no_sync).status);
    // [/snippet:doc-shape-scalar-int32-c]
}

TEST_F(AttributeShapesTest, Float3ArraySingleElement) {
    load_base();

    // Single-element variant of the lanes-based float3[] layout.
    ovx_string_t prim = ovx_str("/World/Plane");
    float point_data[1][3] = {{1.0f, 2.0f, 3.0f}};
    int64_t shape[1] = {1};
    int64_t strides[1] = {1};

    DLTensor write_tensor{};
    write_tensor.data = point_data;
    write_tensor.device = {kDLCPU, 0};
    write_tensor.ndim = 1;
    write_tensor.dtype = {kDLFloat, 32, 3};
    write_tensor.shape = shape;
    write_tensor.strides = strides;

    ovrtx_input_buffer_t buffer{};
    buffer.tensors = &write_tensor;
    buffer.tensor_count = 1;

    ovrtx_binding_desc_or_handle_t binding = ovrtx_make_binding_desc(
        &prim, 1, ovx_str("points"), OVRTX_SEMANTIC_NONE, write_tensor.dtype);
    binding.binding_desc.attribute_type.is_array = true;

    ovrtx_enqueue_result_t eq =
        ovrtx_write_attribute(renderer_, &binding, &buffer, OVRTX_DATA_ACCESS_SYNC);
    ASSERT_API_SUCCESS(eq.status);
    ovrtx_op_wait_result_t wait_result;
    ASSERT_API_SUCCESS(
        ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    // Read back and sanity-check the value.
    ovrtx_read_handle_t read_handle = 0;
    eq = ovrtx_read_attribute(renderer_, &binding, nullptr, &read_handle);
    ASSERT_API_SUCCESS(eq.status);
    ASSERT_API_SUCCESS(
        ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    ovrtx_read_output_t output{};
    ASSERT_API_SUCCESS(
        ovrtx_fetch_read_result(renderer_, read_handle, ovrtx_timeout_infinite, &output).status);

    DLTensor const& t = output.buffers[0].dl;
    EXPECT_EQ(t.ndim, 1);
    EXPECT_EQ(t.shape[0], 1);
    EXPECT_EQ(t.dtype.lanes, 3u);

    float const* data = static_cast<float const*>(t.data);
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);

    ovrtx_cuda_sync_t no_sync{};
    ASSERT_API_SUCCESS(
        ovrtx_release_read_result(renderer_, output.map_handle, no_sync).status);
}

TEST_F(AttributeShapesTest, Float3Array) {
    load_base();

    // [snippet:doc-shape-float3-array-c]
    // point3f[] is a variable-length array of 3-component float vectors.
    // In the C API, express it as a 1-D tensor with shape=[M],
    // dtype={kDLFloat, 32, 3}. The lane count is the vector dimension.
    ovx_string_t prim = ovx_str("/World/Plane");
    float points_data[4][3] = {
        {-50.0f, 0.0f, -50.0f},
        {50.0f, 0.0f, -50.0f},
        {-50.0f, 0.0f, 50.0f},
        {50.0f, 0.0f, 50.0f},
    };
    int64_t shape[1] = {4};
    int64_t strides[1] = {1};

    DLTensor write_tensor{};
    write_tensor.data = points_data;
    write_tensor.device = {kDLCPU, 0};
    write_tensor.ndim = 1;
    write_tensor.dtype = {kDLFloat, 32, 3};
    write_tensor.shape = shape;
    write_tensor.strides = strides;

    ovrtx_input_buffer_t buffer{};
    buffer.tensors = &write_tensor;
    buffer.tensor_count = 1;

    ovrtx_binding_desc_or_handle_t binding = ovrtx_make_binding_desc(
        &prim, 1, ovx_str("points"), OVRTX_SEMANTIC_NONE, write_tensor.dtype);
    binding.binding_desc.attribute_type.is_array = true;

    ovrtx_enqueue_result_t eq =
        ovrtx_write_attribute(renderer_, &binding, &buffer, OVRTX_DATA_ACCESS_SYNC);
    ASSERT_API_SUCCESS(eq.status);
    ovrtx_op_wait_result_t wait_result;
    ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    // Read back — expect ndim=1, shape=[4], dtype.lanes=3.
    ovrtx_read_handle_t read_handle = 0;
    eq = ovrtx_read_attribute(renderer_, &binding, nullptr, &read_handle);
    ASSERT_API_SUCCESS(eq.status);
    ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    ovrtx_read_output_t output{};
    ASSERT_API_SUCCESS(ovrtx_fetch_read_result(renderer_, read_handle, ovrtx_timeout_infinite, &output).status);

    DLTensor const& t = output.buffers[0].dl;
    EXPECT_EQ(t.ndim, 1);
    EXPECT_EQ(t.shape[0], 4);
    EXPECT_EQ(t.dtype.lanes, 3u);

    ovrtx_cuda_sync_t no_sync{};
    ASSERT_API_SUCCESS(ovrtx_release_read_result(renderer_, output.map_handle, no_sync).status);
    // [/snippet:doc-shape-float3-array-c]
}

TEST_F(AttributeShapesTest, Mat4Array) {
    load_base();

    // [snippet:doc-shape-mat4-array-c]
    // A per-prim 4x4 matrix attribute is a 1-D tensor with shape=[N],
    // dtype={kDLFloat, 64, 16}. The lane count is the matrix element count.
    ovx_string_t prim = ovx_str("/World/Camera");
    double transforms[1][16] = {{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        10.0, 20.0, 30.0, 1.0,
    }};
    int64_t shape[1] = {1};
    int64_t strides[1] = {1};

    DLTensor write_tensor{};
    write_tensor.data = transforms;
    write_tensor.device = {kDLCPU, 0};
    write_tensor.ndim = 1;
    write_tensor.dtype = {kDLFloat, 64, 16};
    write_tensor.shape = shape;
    write_tensor.strides = strides;

    ovrtx_input_buffer_t buffer{};
    buffer.tensors = &write_tensor;
    buffer.tensor_count = 1;

    ovrtx_binding_desc_or_handle_t binding = ovrtx_make_binding_desc(
        &prim, 1, ovx_str("omni:xform"), OVRTX_SEMANTIC_XFORM_MAT4x4, write_tensor.dtype);

    ovrtx_enqueue_result_t eq =
        ovrtx_write_attribute(renderer_, &binding, &buffer, OVRTX_DATA_ACCESS_SYNC);
    ASSERT_API_SUCCESS(eq.status);
    ovrtx_op_wait_result_t wait_result;
    ASSERT_API_SUCCESS(
        ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    // Read back — expect ndim=1, shape=[1], dtype.lanes=16.
    ovrtx_binding_desc_or_handle_t read_binding = ovrtx_make_binding_desc(
        &prim, 1, ovx_str("omni:xform"), OVRTX_SEMANTIC_NONE, write_tensor.dtype);
    ovrtx_read_handle_t read_handle = 0;
    eq = ovrtx_read_attribute(renderer_, &read_binding, nullptr, &read_handle);
    ASSERT_API_SUCCESS(eq.status);
    ASSERT_API_SUCCESS(
        ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    ovrtx_read_output_t output{};
    ASSERT_API_SUCCESS(
        ovrtx_fetch_read_result(renderer_, read_handle, ovrtx_timeout_infinite, &output).status);

    DLTensor const& t = output.buffers[0].dl;
    EXPECT_EQ(t.ndim, 1);
    EXPECT_EQ(t.shape[0], 1);
    EXPECT_EQ(t.dtype.lanes, 16u);

    double const* data = static_cast<double const*>(t.data);
    // Row 3 (last row) holds the translation.
    EXPECT_DOUBLE_EQ(data[3 * 4 + 0], 10.0);
    EXPECT_DOUBLE_EQ(data[3 * 4 + 1], 20.0);
    EXPECT_DOUBLE_EQ(data[3 * 4 + 2], 30.0);

    ovrtx_cuda_sync_t no_sync{};
    ASSERT_API_SUCCESS(ovrtx_release_read_result(renderer_, output.map_handle, no_sync).status);
    // [/snippet:doc-shape-mat4-array-c]
}
