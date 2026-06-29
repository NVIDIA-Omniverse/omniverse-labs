// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Tests for C code examples in camera_sensors.rst

#include <gtest/gtest.h>
#include "helpers.h"

#include <cstring>
#include <string>

TEST(CameraSensors, StepAndMapOutputs) {
    // Create renderer
    TestConfig tc("CameraSensors");
    ovrtx_renderer_t* renderer = nullptr;
    ovrtx_result_t result = ovrtx_create_renderer(&tc.config, &renderer);
    ASSERT_API_SUCCESS(result.status);

    // Load scene via sublayer
    std::string scene_path = get_test_data_dir() + "/simple_camera.usda";
    std::string usda = make_sublayer_usda(scene_path, R"usda(
def "Render" {
    def RenderProduct "Camera" {
        int2 resolution = (640, 480)
        rel camera = </Camera0>
        rel orderedVars = [<LdrColor>, <HdrColor>]

        def RenderVar "LdrColor" {
            string sourceName = "LdrColor"
        }

        def RenderVar "HdrColor" {
            string sourceName = "HdrColor"
        }
    }
}
)usda");

    ovrtx_enqueue_result_t enqueue_result = ovrtx_open_usd_from_string(renderer, {usda.c_str(), usda.size()});
    ASSERT_API_SUCCESS(enqueue_result.status);

    ovrtx_op_wait_result_t wait_result;
    result = ovrtx_wait_op(renderer, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    // Step renderer
    ovx_string_t rp_path = ovx_str("/Render/Camera");
    ovrtx_render_product_set_t render_products{};
    render_products.render_products = &rp_path;
    render_products.num_render_products = 1;

    ovrtx_step_result_handle_t step_handle = 0;
    enqueue_result = ovrtx_step(renderer, render_products, 1.0 / 60.0, &step_handle);
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    // Fetch results
    ovrtx_render_product_set_outputs_t outputs{};
    result = ovrtx_fetch_results(renderer, step_handle, ovrtx_timeout_infinite, &outputs);
    ASSERT_API_SUCCESS(result.status);

    // [snippet:doc-step-and-map-camera-outputs-c]
    ovrtx_render_var_output_handle_t ldr_handle = find_output(outputs, "LdrColor");
    ovrtx_render_var_output_handle_t hdr_handle = find_output(outputs, "HdrColor");
    ASSERT_NE(ldr_handle, OVRTX_INVALID_HANDLE);
    ASSERT_NE(hdr_handle, OVRTX_INVALID_HANDLE);

    // Map LdrColor to CPU
    ovrtx_map_output_description_t map_desc = {};
    map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;

    ovrtx_render_var_output_t ldr_output = {};
    result = ovrtx_map_render_var_output(renderer, ldr_handle, &map_desc,
                                         ovrtx_timeout_infinite, &ldr_output);
    ASSERT_API_SUCCESS(result.status);

    DLTensor const& ldr_tensor = *ldr_output.tensors[0].dl;
    // ldr_tensor.data: RGBA uint8 pixels
    // ldr_tensor.shape[0]: height, ldr_tensor.shape[1]: width

    save_ldr_png("CameraSensors.LdrColor",
                 ldr_tensor.data,
                 static_cast<int>(ldr_tensor.shape[1]),
                 static_cast<int>(ldr_tensor.shape[0]));

    ovrtx_cuda_sync_t no_sync = {};
    ovrtx_unmap_render_var_output(renderer, ldr_output.map_handle, no_sync);
    // [/snippet:doc-step-and-map-camera-outputs-c]

    // [snippet:doc-map-render-output-cuda-c]
    map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CUDA;
    map_desc.sync_stream = 1; // CUDA default stream
    ovrtx_render_var_output_t cuda_output = {};
    result = ovrtx_map_render_var_output(renderer, ldr_handle, &map_desc,
                                         ovrtx_timeout_infinite, &cuda_output);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_EQ(cuda_output.tensors[0].dl->device.device_type, kDLCUDA);
    ovrtx_unmap_render_var_output(renderer, cuda_output.map_handle, no_sync);
    // [/snippet:doc-map-render-output-cuda-c]

    // [snippet:doc-map-render-output-cuda-array-c]
    map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY;
    map_desc.sync_stream = 1; // CUDA default stream
    ovrtx_render_var_output_t cuda_array_output = {};
    result = ovrtx_map_render_var_output(renderer, ldr_handle, &map_desc,
                                         ovrtx_timeout_infinite, &cuda_array_output);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_EQ(cuda_array_output.tensors[0].dl->device.device_type, kDLCUDA);
    ASSERT_EQ(cuda_array_output.tensors[0].dl->dtype.code, kDLOpaqueHandle);
    ASSERT_NE(cuda_array_output.tensors[0].dl->data, nullptr);
    ovrtx_unmap_render_var_output(renderer, cuda_array_output.map_handle, no_sync);
    // [/snippet:doc-map-render-output-cuda-array-c]

    // Map HdrColor to CPU to verify it too
    map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;
    map_desc.sync_stream = 0;
    ovrtx_render_var_output_t hdr_output = {};
    result = ovrtx_map_render_var_output(renderer, hdr_handle, &map_desc,
                                       ovrtx_timeout_infinite, &hdr_output);
    ASSERT_API_SUCCESS(result.status);

    DLTensor const& hdr_tensor = *hdr_output.tensors[0].dl;
    EXPECT_EQ(hdr_tensor.shape[0], 480);
    EXPECT_EQ(hdr_tensor.shape[1], 640);

    ovrtx_unmap_render_var_output(renderer, hdr_output.map_handle, no_sync);

    // Cleanup
    ovrtx_destroy_results(renderer, step_handle);
    ovrtx_destroy_renderer(renderer);
}
