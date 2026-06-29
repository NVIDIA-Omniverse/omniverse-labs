// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Tests using ovrtx-test-base.usda

#include <gtest/gtest.h>
#include "helpers.h"

#include <ovrtx/ovrtx_attributes.h>

#include <cmath>
#include <cstring>
#include <string>

class BaseTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        TestConfig tc("BaseTest");
        ovrtx_result_t result = ovrtx_create_renderer(&tc.config, &renderer_);
        ASSERT_API_SUCCESS(result.status);
    }

    static void TearDownTestSuite() {
        if (renderer_) {
            ovrtx_destroy_renderer(renderer_);
            renderer_ = nullptr;
        }
    }

    static ovrtx_renderer_t* renderer_;
};

ovrtx_renderer_t* BaseTest::renderer_ = nullptr;

TEST_F(BaseTest, RenderLdrColor) {
    // Reset to a clean state and load the test base scene
    ovrtx_op_wait_result_t wait_result;
    ovrtx_result_t result;
    ovrtx_enqueue_result_t enqueue_result = ovrtx_reset_stage(renderer_);
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    std::string scene_path = get_docs_test_data_dir() + "/ovrtx-test-base.usda";
    enqueue_result = ovrtx_open_usd_from_file(renderer_, {scene_path.c_str(), scene_path.size()});
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    enqueue_result = ovrtx_reset(renderer_, 0.0);
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    // Step renderer (warm up + render)
    ovx_string_t rp_path = ovx_str("/Render/Camera");
    ovrtx_render_product_set_t render_products{};
    render_products.render_products = &rp_path;
    render_products.num_render_products = 1;

    ovrtx_step_result_handle_t step_handle = 0;
    for (int i = 0; i < 40; ++i) {
        enqueue_result = ovrtx_step(renderer_, render_products, 1.0 / 60.0, &step_handle);
        ASSERT_API_SUCCESS(enqueue_result.status);
        result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
        ASSERT_API_SUCCESS(result.status);
        ASSERT_NO_OP_ERRORS(wait_result);
        ovrtx_destroy_results(renderer_, step_handle);
    }

    // Render one frame
    enqueue_result = ovrtx_step(renderer_, render_products, 1.0 / 60.0, &step_handle);
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    // Fetch results
    ovrtx_render_product_set_outputs_t outputs{};
    result = ovrtx_fetch_results(renderer_, step_handle, ovrtx_timeout_infinite, &outputs);
    ASSERT_API_SUCCESS(result.status);

    // Find and map LdrColor
    ovrtx_render_var_output_handle_t ldr_handle = find_output(outputs, "LdrColor");
    ASSERT_NE(ldr_handle, OVRTX_INVALID_HANDLE);

    ovrtx_map_output_description_t map_desc = {};
    map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;

    ovrtx_render_var_output_t ldr_output = {};
    result = ovrtx_map_render_var_output(renderer_, ldr_handle, &map_desc,
                                       ovrtx_timeout_infinite, &ldr_output);
    ASSERT_API_SUCCESS(result.status);

    DLTensor const& tensor = *ldr_output.tensors[0].dl;
    EXPECT_GT(tensor.shape[0], 0);
    EXPECT_GT(tensor.shape[1], 0);

    save_ldr_png("base.Camera.LdrColor.0001",
                 tensor.data,
                 static_cast<int>(tensor.shape[1]),
                 static_cast<int>(tensor.shape[0]));

    ovrtx_cuda_sync_t no_sync = {};
    ovrtx_unmap_render_var_output(renderer_, ldr_output.map_handle, no_sync);

    ovrtx_destroy_results(renderer_, step_handle);
}

TEST_F(BaseTest, BindMaterial) {
    // Reset to a clean state and load the test base scene
    ovrtx_op_wait_result_t wait_result;
    ovrtx_result_t result;
    ovrtx_enqueue_result_t enqueue_result = ovrtx_reset_stage(renderer_);
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    std::string scene_path = get_docs_test_data_dir() + "/ovrtx-test-base.usda";
    enqueue_result = ovrtx_open_usd_from_file(renderer_, {scene_path.c_str(), scene_path.size()});
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    enqueue_result = ovrtx_reset(renderer_, 0.0);
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    // [snippet:doc-bind-material-c]
    // Bind /World/Looks/srf_glass to /World/logo
    ovx_string_t prim_path = ovx_str("/World/logo/logo/logo");
    ovx_string_t material_path = ovx_str("/World/Looks/srf_glass");
    enqueue_result = ovrtx_set_path_attributes(
        renderer_, &prim_path, 1,
        ovx_str("material:binding"), &material_path);
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);
    // [/snippet:doc-bind-material-c]

    // [snippet:doc-warmup-c]
    // Warm up - step enough frames for texture streaming to finish loading
    // high-res mips and for path tracing to converge to a good quality image.
    ovx_string_t rp_path = ovx_str("/Render/Camera");
    ovrtx_render_product_set_t render_products{};
    render_products.render_products = &rp_path;
    render_products.num_render_products = 1;

    ovrtx_step_result_handle_t step_handle = 0;
    int const WARMUP_FRAMES = 40;
    for (int i = 0; i < WARMUP_FRAMES; ++i) {
        enqueue_result = ovrtx_step(renderer_, render_products, 1.0 / 60.0, &step_handle);
        ASSERT_API_SUCCESS(enqueue_result.status);
        result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
        ASSERT_API_SUCCESS(result.status);
        ASSERT_NO_OP_ERRORS(wait_result);
        ovrtx_destroy_results(renderer_, step_handle);
    }
    // [/snippet:doc-warmup-c]

    // Render one frame
    enqueue_result = ovrtx_step(renderer_, render_products, 1.0 / 60.0, &step_handle);
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    // Fetch results
    ovrtx_render_product_set_outputs_t outputs{};
    result = ovrtx_fetch_results(renderer_, step_handle, ovrtx_timeout_infinite, &outputs);
    ASSERT_API_SUCCESS(result.status);

    // Find and map LdrColor
    ovrtx_render_var_output_handle_t ldr_handle = find_output(outputs, "LdrColor");
    ASSERT_NE(ldr_handle, OVRTX_INVALID_HANDLE);

    ovrtx_map_output_description_t map_desc = {};
    map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;

    ovrtx_render_var_output_t ldr_output = {};
    result = ovrtx_map_render_var_output(renderer_, ldr_handle, &map_desc,
                                       ovrtx_timeout_infinite, &ldr_output);
    ASSERT_API_SUCCESS(result.status);

    DLTensor const& tensor = *ldr_output.tensors[0].dl;
    EXPECT_GT(tensor.shape[0], 0);
    EXPECT_GT(tensor.shape[1], 0);

    save_ldr_png("bind_material.Camera.LdrColor.0001",
                 tensor.data,
                 static_cast<int>(tensor.shape[1]),
                 static_cast<int>(tensor.shape[0]));

    ovrtx_cuda_sync_t no_sync = {};
    ovrtx_unmap_render_var_output(renderer_, ldr_output.map_handle, no_sync);

    ovrtx_destroy_results(renderer_, step_handle);
}

TEST_F(BaseTest, SettingsRtptMaxBounces) {
    // Reset to a clean state and load the test base scene
    ovrtx_op_wait_result_t wait_result;
    ovrtx_result_t result;
    ovrtx_enqueue_result_t enqueue_result = ovrtx_reset_stage(renderer_);
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    std::string scene_path = get_docs_test_data_dir() + "/ovrtx-test-base.usda";
    enqueue_result = ovrtx_open_usd_from_file(renderer_, {scene_path.c_str(), scene_path.size()});
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    int bounce_values[] = {2, 3, 23};
    for (int max_bounces : bounce_values) {
        // [snippet:doc-set-render-setting-c]
        // Set a render setting on the RenderProduct prim
        ovx_string_t rp = ovx_str("/Render/Camera");
        int32_t value = max_bounces;
        size_t count = 1;

        DLDataType int32_type = {kDLInt, 32, 1};
        DLTensor tensor = ovrtx_make_write_cpu_tensor(&value, &count, int32_type);

        ovrtx_input_buffer_t buffer{};
        buffer.tensors = &tensor;
        buffer.tensor_count = 1;

        ovrtx_binding_desc_or_handle_t binding =
            ovrtx_make_binding_desc(&rp, 1, ovx_str("omni:rtx:rtpt:maxBounces"),
                                    OVRTX_SEMANTIC_NONE, int32_type);

        enqueue_result = ovrtx_write_attribute(renderer_, &binding, &buffer, OVRTX_DATA_ACCESS_SYNC);
        ASSERT_API_SUCCESS(enqueue_result.status);
        result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
        ASSERT_API_SUCCESS(result.status);
        ASSERT_NO_OP_ERRORS(wait_result);
        // [/snippet:doc-set-render-setting-c]

        // Reset and warm up so the setting takes effect
        enqueue_result = ovrtx_reset(renderer_, 0.0);
        ASSERT_API_SUCCESS(enqueue_result.status);
        result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
        ASSERT_API_SUCCESS(result.status);
        ASSERT_NO_OP_ERRORS(wait_result);

        ovx_string_t rp_path = ovx_str("/Render/Camera");
        ovrtx_render_product_set_t render_products{};
        render_products.render_products = &rp_path;
        render_products.num_render_products = 1;

        ovrtx_step_result_handle_t step_handle = 0;
        for (int i = 0; i < 40; ++i) {
            enqueue_result = ovrtx_step(renderer_, render_products, 1.0 / 60.0, &step_handle);
            ASSERT_API_SUCCESS(enqueue_result.status);
            result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
            ASSERT_API_SUCCESS(result.status);
            ASSERT_NO_OP_ERRORS(wait_result);
            ovrtx_destroy_results(renderer_, step_handle);
        }

        // Render one frame
        enqueue_result = ovrtx_step(renderer_, render_products, 1.0 / 60.0, &step_handle);
        ASSERT_API_SUCCESS(enqueue_result.status);
        result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
        ASSERT_API_SUCCESS(result.status);
        ASSERT_NO_OP_ERRORS(wait_result);

        // Fetch results
        ovrtx_render_product_set_outputs_t outputs{};
        result = ovrtx_fetch_results(renderer_, step_handle, ovrtx_timeout_infinite, &outputs);
        ASSERT_API_SUCCESS(result.status);

        // Find and map LdrColor
        ovrtx_render_var_output_handle_t ldr_handle = find_output(outputs, "LdrColor");
        ASSERT_NE(ldr_handle, OVRTX_INVALID_HANDLE);

        ovrtx_map_output_description_t map_desc = {};
        map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;

        ovrtx_render_var_output_t ldr_output = {};
        result = ovrtx_map_render_var_output(renderer_, ldr_handle, &map_desc,
                                           ovrtx_timeout_infinite, &ldr_output);
        ASSERT_API_SUCCESS(result.status);

        DLTensor const& out_tensor = *ldr_output.tensors[0].dl;
        EXPECT_GT(out_tensor.shape[0], 0);
        EXPECT_GT(out_tensor.shape[1], 0);

        std::string name = "settings_rtpt_maxBounces.Camera.LdrColor.maxBounces-"
                           + std::to_string(max_bounces) + ".0001";
        save_ldr_png(name.c_str(),
                     out_tensor.data,
                     static_cast<int>(out_tensor.shape[1]),
                     static_cast<int>(out_tensor.shape[0]));

        ovrtx_cuda_sync_t no_sync = {};
        ovrtx_unmap_render_var_output(renderer_, ldr_output.map_handle, no_sync);

        ovrtx_destroy_results(renderer_, step_handle);
    }
}

TEST_F(BaseTest, SettingsRtptMaxSpecularAndTransmissionBounces) {
    // Reset to a clean state and load the test base scene
    ovrtx_op_wait_result_t wait_result;
    ovrtx_result_t result;
    ovrtx_enqueue_result_t enqueue_result = ovrtx_reset_stage(renderer_);
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    std::string scene_path = get_docs_test_data_dir() + "/ovrtx-test-base.usda";
    enqueue_result = ovrtx_open_usd_from_file(renderer_, {scene_path.c_str(), scene_path.size()});
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    // Bind glass material to the logo so specular/transmission bounces are visible
    ovx_string_t prim_path = ovx_str("/World/logo/logo/logo");
    ovx_string_t material_path = ovx_str("/World/Looks/srf_glass");
    enqueue_result = ovrtx_set_path_attributes(
        renderer_, &prim_path, 1,
        ovx_str("material:binding"), &material_path);
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    int bounce_values[] = {2, 3, 23};
    for (int bounces : bounce_values) {
        // Set maxSpecularAndTransmissionBounces on the RenderProduct prim
        ovx_string_t rp = ovx_str("/Render/Camera");
        int32_t value = bounces;
        size_t count = 1;

        DLDataType int32_type = {kDLInt, 32, 1};
        DLTensor tensor = ovrtx_make_write_cpu_tensor(&value, &count, int32_type);

        ovrtx_input_buffer_t buffer{};
        buffer.tensors = &tensor;
        buffer.tensor_count = 1;

        ovrtx_binding_desc_or_handle_t binding =
            ovrtx_make_binding_desc(&rp, 1,
                                    ovx_str("omni:rtx:rtpt:maxSpecularAndTransmissionBounces"),
                                    OVRTX_SEMANTIC_NONE, int32_type);

        enqueue_result = ovrtx_write_attribute(renderer_, &binding, &buffer, OVRTX_DATA_ACCESS_SYNC);
        ASSERT_API_SUCCESS(enqueue_result.status);
        result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
        ASSERT_API_SUCCESS(result.status);
        ASSERT_NO_OP_ERRORS(wait_result);

        // Reset and warm up so the setting takes effect
        enqueue_result = ovrtx_reset(renderer_, 0.0);
        ASSERT_API_SUCCESS(enqueue_result.status);
        result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
        ASSERT_API_SUCCESS(result.status);
        ASSERT_NO_OP_ERRORS(wait_result);

        ovx_string_t rp_path = ovx_str("/Render/Camera");
        ovrtx_render_product_set_t render_products{};
        render_products.render_products = &rp_path;
        render_products.num_render_products = 1;

        ovrtx_step_result_handle_t step_handle = 0;
        for (int i = 0; i < 40; ++i) {
            enqueue_result = ovrtx_step(renderer_, render_products, 1.0 / 60.0, &step_handle);
            ASSERT_API_SUCCESS(enqueue_result.status);
            result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
            ASSERT_API_SUCCESS(result.status);
            ASSERT_NO_OP_ERRORS(wait_result);
            ovrtx_destroy_results(renderer_, step_handle);
        }

        // Render one frame
        enqueue_result = ovrtx_step(renderer_, render_products, 1.0 / 60.0, &step_handle);
        ASSERT_API_SUCCESS(enqueue_result.status);
        result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
        ASSERT_API_SUCCESS(result.status);
        ASSERT_NO_OP_ERRORS(wait_result);

        // Fetch results
        ovrtx_render_product_set_outputs_t outputs{};
        result = ovrtx_fetch_results(renderer_, step_handle, ovrtx_timeout_infinite, &outputs);
        ASSERT_API_SUCCESS(result.status);

        // Find and map LdrColor
        ovrtx_render_var_output_handle_t ldr_handle = find_output(outputs, "LdrColor");
        ASSERT_NE(ldr_handle, OVRTX_INVALID_HANDLE);

        ovrtx_map_output_description_t map_desc = {};
        map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;

        ovrtx_render_var_output_t ldr_output = {};
        result = ovrtx_map_render_var_output(renderer_, ldr_handle, &map_desc,
                                           ovrtx_timeout_infinite, &ldr_output);
        ASSERT_API_SUCCESS(result.status);

        DLTensor const& out_tensor = *ldr_output.tensors[0].dl;
        EXPECT_GT(out_tensor.shape[0], 0);
        EXPECT_GT(out_tensor.shape[1], 0);

        std::string name = "settings_rtpt_maxSpecularAndTransmissionBounces.Camera.LdrColor.maxSpecularAndTransmissionBounces-"
                           + std::to_string(bounces) + ".0001";
        save_ldr_png(name.c_str(),
                     out_tensor.data,
                     static_cast<int>(out_tensor.shape[1]),
                     static_cast<int>(out_tensor.shape[0]));

        ovrtx_cuda_sync_t no_sync = {};
        ovrtx_unmap_render_var_output(renderer_, ldr_output.map_handle, no_sync);

        ovrtx_destroy_results(renderer_, step_handle);
    }
}


TEST_F(BaseTest, UpdateFromUsdTimeAsync) {
    // Asynchronously evaluate a time-sampled attribute at two distinct times.
    //
    // The animated sublayer authors xformOp:translate time samples on /World/logo
    // spanning X=-100 at timecode 0 to X=+100 at timecode 60, with
    // timeCodesPerSecond=60 (so timecode 60 corresponds to 1 second).
    ovrtx_op_wait_result_t wait_result;
    ovrtx_result_t result;
    ovrtx_enqueue_result_t enqueue_result = ovrtx_reset_stage(renderer_);
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    std::string scene_path = get_docs_test_data_dir() + "/ovrtx-test-base-logo-animated.usda";

    enqueue_result = ovrtx_open_usd_from_file(renderer_, {scene_path.c_str(), scene_path.size()});
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    enqueue_result = ovrtx_reset(renderer_, 0.0);
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    ovx_string_t rp_path = ovx_str("/Render/Camera");
    ovrtx_render_product_set_t render_products{};
    render_products.render_products = &rp_path;
    render_products.num_render_products = 1;

    auto translate_x_at = [&](double t_seconds, double* out_x) {
        ovrtx_op_wait_result_t wr;
        ovrtx_result_t r;
        ovrtx_enqueue_result_t eq;

        // [snippet:doc-update-from-usd-time-async-c]
        // ovrtx_update_stage_from_usd_time takes USD time; the runtime
        // re-evaluates time-sampled attributes against the stage's
        // timeCodesPerSecond metadata.
        eq = ovrtx_update_stage_from_usd_time(renderer_, t_seconds);
        ASSERT_API_SUCCESS(eq.status);
        r = ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wr);
        ASSERT_API_SUCCESS(r.status);
        // [/snippet:doc-update-from-usd-time-async-c]

        // Drive the render pipeline once so the time-sampled attribute writes
        // are consumed before we read the composed transform back (mirrors
        // the step() between each update in the Python equivalent).
        ovrtx_step_result_handle_t step_handle = 0;
        eq = ovrtx_step(renderer_, render_products, 1.0, &step_handle);
        ASSERT_API_SUCCESS(eq.status);
        r = ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wr);
        ASSERT_API_SUCCESS(r.status);
        ovrtx_destroy_results(renderer_, step_handle);

        // Read the composed local matrix back. In the C API, omni:xform is a
        // 1-D tensor with shape=[N] and dtype.lanes=16, one 4x4 matrix per prim.
        ovx_string_t prim = ovx_str("/World/logo");
        DLDataType matrix_type = {kDLFloat, 64, 16};
        ovrtx_binding_desc_or_handle_t binding = ovrtx_make_binding_desc(
            &prim, 1, ovx_str("omni:xform"), OVRTX_SEMANTIC_NONE, matrix_type);

        ovrtx_read_handle_t read_handle = 0;
        eq = ovrtx_read_attribute(renderer_, &binding, nullptr, &read_handle);
        ASSERT_API_SUCCESS(eq.status);
        r = ovrtx_wait_op(renderer_, eq.op_index, ovrtx_timeout_infinite, &wr);
        ASSERT_API_SUCCESS(r.status);

        ovrtx_read_output_t output{};
        r = ovrtx_fetch_read_result(renderer_, read_handle, ovrtx_timeout_infinite, &output);
        ASSERT_API_SUCCESS(r.status);
        ASSERT_EQ(output.buffer_count, 1u);
        ASSERT_NE(output.buffers, nullptr);

        DLTensor const& t = output.buffers[0].dl;
        ASSERT_EQ(t.ndim, 1);
        ASSERT_NE(t.shape, nullptr);
        ASSERT_EQ(t.shape[0], 1);
        ASSERT_EQ(t.dtype.lanes, 16u);
        ASSERT_NE(t.data, nullptr);
        double const* data = static_cast<double const*>(t.data);
        // USD row-vector convention: translation lives in row 3, cols 0..2.
        *out_x = data[3 * 4 + 0];

        ovrtx_cuda_sync_t no_sync{};
        ovrtx_release_read_result(renderer_, output.map_handle, no_sync);
    };

    // 0s -> timecode 0 -> X=-100; 1s -> timecode 60 -> X=+100.
    double x_at_start = 0.0;
    double x_at_end = 0.0;
    translate_x_at(0.0, &x_at_start);
    if (HasFatalFailure()) {
        return;
    }
    translate_x_at(1.0, &x_at_end);
    if (HasFatalFailure()) {
        return;
    }

    EXPECT_GT(std::fabs(x_at_end - x_at_start), 1.0)
        << "expected time-sampled translate to change across time, "
        << "got x(0s)=" << x_at_start << ", x(1s)=" << x_at_end;
}
