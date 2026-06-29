// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Tests for ovrtx_read_attribute / ovrtx_fetch_read_result / ovrtx_release_read_result.
// Scalar test: write-then-read omni:rtx:rtpt:maxBounces on /Render/Camera.
// Array test: read the authored "points" attribute from /World/Plane.

#include <gtest/gtest.h>
#include "helpers.h"

#include <ovrtx/ovrtx_attributes.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

class AttributeReadTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        TestConfig tc("AttributeReadTest");
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

    // Write a single int32 value to the given prim/attribute so the subsequent
    // read has a deterministic value to observe.
    static void write_int32(char const* prim, char const* attribute, int32_t value) {
        ovx_string_t p = ovx_str(prim);
        size_t count = 1;
        DLDataType int32_type = {kDLInt, 32, 1};
        DLTensor tensor = ovrtx_make_write_cpu_tensor(&value, &count, int32_type);

        ovrtx_input_buffer_t buffer{};
        buffer.tensors = &tensor;
        buffer.tensor_count = 1;

        ovrtx_binding_desc_or_handle_t binding = ovrtx_make_binding_desc(
            &p, 1, ovx_str(attribute), OVRTX_SEMANTIC_NONE, int32_type);

        ovrtx_enqueue_result_t eq =
            ovrtx_write_attribute(renderer_, &binding, &buffer, OVRTX_DATA_ACCESS_SYNC);
        ASSERT_API_SUCCESS(eq.status);
        ovrtx_op_wait_result_t wait_result;
        ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
        ASSERT_NO_OP_ERRORS(wait_result);
    }

    static ovrtx_renderer_t* renderer_;
};

ovrtx_renderer_t* AttributeReadTest::renderer_ = nullptr;

TEST_F(AttributeReadTest, ReadScalarAttribute) {
    load_base();
    write_int32("/Render/Camera", "omni:rtx:rtpt:maxBounces", 17);

    // [snippet:doc-read-attribute-scalar-c]
    // Describe the attribute to read: one prim, name, element type matching
    // how the runtime stores it (maxBounces is a 32-bit unsigned integer).
    ovx_string_t rp = ovx_str("/Render/Camera");
    DLDataType uint32_type = {kDLUInt, 32, 1};
    ovrtx_binding_desc_or_handle_t binding = ovrtx_make_binding_desc(
        &rp, 1, ovx_str("omni:rtx:rtpt:maxBounces"), OVRTX_SEMANTIC_NONE, uint32_type);

    // Enqueue the read. Pass NULL for read_dest so ovrtx allocates internal storage.
    ovrtx_read_handle_t read_handle = 0;
    ovrtx_enqueue_result_t eq = ovrtx_read_attribute(renderer_, &binding, nullptr, &read_handle);
    ASSERT_API_SUCCESS(eq.status);

    ovrtx_op_wait_result_t wait_result;
    ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    // Fetch the DLPack tensor(s). Scalar reads produce buffer_count == 1 with
    // shape [prim_count].
    ovrtx_read_output_t output{};
    ASSERT_API_SUCCESS(ovrtx_fetch_read_result(renderer_, read_handle, ovrtx_timeout_infinite, &output).status);
    ASSERT_EQ(output.buffer_count, 1u);
    ASSERT_EQ(output.prim_count, 1u);

    DLTensor const& t = output.buffers[0].dl;
    uint32_t value = *static_cast<uint32_t const*>(t.data);

    // Release the read result when done. The output pointers are invalidated.
    ovrtx_cuda_sync_t no_sync{};
    ASSERT_API_SUCCESS(ovrtx_release_read_result(renderer_, output.map_handle, no_sync).status);
    // [/snippet:doc-read-attribute-scalar-c]

    EXPECT_EQ(value, 17u);
}

TEST_F(AttributeReadTest, ReadArrayAttribute) {
    load_base();

    // [snippet:doc-read-array-attribute-c]
    // Array attributes are variable-length per prim. The dtype in the binding
    // is (code, bits, lanes) — lanes expresses multi-component element types.
    // `points` is float3[], so request float32 with lanes=3: one element per
    // point, three lanes per point. Override is_array=true (the helper default
    // is false).
    ovx_string_t prim = ovx_str("/World/Plane");
    DLDataType point3f_type = {kDLFloat, 32, 3};
    ovrtx_binding_desc_or_handle_t binding = ovrtx_make_binding_desc(
        &prim, 1, ovx_str("points"), OVRTX_SEMANTIC_NONE, point3f_type);
    binding.binding_desc.attribute_type.is_array = true;

    ovrtx_read_handle_t read_handle = 0;
    ovrtx_enqueue_result_t eq = ovrtx_read_attribute(renderer_, &binding, nullptr, &read_handle);
    ASSERT_API_SUCCESS(eq.status);

    ovrtx_op_wait_result_t wait_result;
    ASSERT_API_SUCCESS(ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wait_result).status);
    ASSERT_NO_OP_ERRORS(wait_result);

    // For arrays the output has one tensor per prim (buffer_count == prim_count).
    ovrtx_read_output_t output{};
    ASSERT_API_SUCCESS(ovrtx_fetch_read_result(renderer_, read_handle, ovrtx_timeout_infinite, &output).status);
    ASSERT_TRUE(output.is_array);
    ASSERT_EQ(output.prim_count, 1u);
    ASSERT_EQ(output.buffer_count, 1u);

    DLTensor const& t = output.buffers[0].dl;
    // The Plane has 4 float3 points. The C API returns lane-based attribute
    // tensors, so this is shape=[4], dtype={kDLFloat, 32, 3}.
    ASSERT_EQ(t.ndim, 1);
    ASSERT_EQ(t.shape[0], 4);
    ASSERT_EQ(t.dtype.code, kDLFloat);
    ASSERT_EQ(t.dtype.bits, 32u);
    ASSERT_EQ(t.dtype.lanes, 3u);
    int64_t element_count = t.shape[0] * t.dtype.lanes;

    ovrtx_cuda_sync_t no_sync{};
    ASSERT_API_SUCCESS(ovrtx_release_read_result(renderer_, output.map_handle, no_sync).status);
    // [/snippet:doc-read-array-attribute-c]

    EXPECT_EQ(element_count, 12);
}
