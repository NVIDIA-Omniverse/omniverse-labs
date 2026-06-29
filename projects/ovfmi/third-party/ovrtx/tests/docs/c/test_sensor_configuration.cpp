// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Tests for C code examples in sensor_configuration.rst

#include <gtest/gtest.h>
#include "helpers.h"

#include <cstring>
#include <string>

static void assert_wait_ok(ovrtx_renderer_t* renderer,
                           ovrtx_op_wait_result_t const& wait_result,
                           char const* context) {
    if (wait_result.num_error_ops == 0)
        return;
    for (size_t i = 0; i < wait_result.num_error_ops; ++i) {
        ovx_string_t msg = ovrtx_get_last_op_error(wait_result.error_op_ids[i]);
        ADD_FAILURE() << context << ": op " << wait_result.error_op_ids[i]
                      << " failed: " << std::string(msg.ptr, msg.length);
    }
}

TEST(SensorConfiguration, StepMultipleRenderProducts) {
    TestConfig tc("SensorConfiguration.StepMultipleRenderProducts");
    ovrtx_renderer_t* renderer = nullptr;
    ovrtx_result_t result = ovrtx_create_renderer(&tc.config, &renderer);
    ASSERT_API_SUCCESS(result.status);

    std::string scene_path = get_test_data_dir() + "/simple_camera.usda";
    std::string usda = make_sublayer_usda(scene_path, R"usda(
def "Render" {
    def RenderProduct "FrontCamera" {
        int2 resolution = (640, 480)
        rel camera = </Camera0>
        rel orderedVars = [<../Vars/LdrColor>]
    }

    def RenderProduct "RearCamera" {
        int2 resolution = (640, 480)
        rel camera = </Camera1>
        rel orderedVars = [<../Vars/LdrColor>]
    }

    def "Vars" {
        def RenderVar "LdrColor" {
            string sourceName = "LdrColor"
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

    // [snippet:doc-step-multiple-render-products-c]
    ovx_string_t rp_paths[] = {
        {"/Render/FrontCamera", strlen("/Render/FrontCamera")},
        {"/Render/RearCamera",  strlen("/Render/RearCamera")},
    };
    ovrtx_render_product_set_t render_products = {};
    render_products.render_products = rp_paths;
    render_products.num_render_products = 2;

    ovrtx_step_result_handle_t step_handle = 0;
    enqueue_result = ovrtx_step(renderer, render_products, 1.0 / 60.0, &step_handle);
    // [/snippet:doc-step-multiple-render-products-c]
    ASSERT_API_SUCCESS(enqueue_result.status);

    result = ovrtx_wait_op(renderer, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);
    ovrtx_destroy_results(renderer, step_handle);

    enqueue_result = ovrtx_step(renderer, render_products, 1.0 / 60.0, &step_handle);
    ASSERT_API_SUCCESS(enqueue_result.status);
    result = ovrtx_wait_op(renderer, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);

    // Fetch and verify we got outputs for both products
    ovrtx_render_product_set_outputs_t outputs{};
    result = ovrtx_fetch_results(renderer, step_handle, ovrtx_timeout_infinite, &outputs);
    ASSERT_API_SUCCESS(result.status);
    EXPECT_EQ(outputs.output_count, 2u);

    // Save LdrColor from each render product
    char const* product_names[] = {"FrontCamera", "RearCamera"};
    ovrtx_map_output_description_t map_desc = {};
    map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;
    ovrtx_cuda_sync_t no_sync = {};
    for (size_t i = 0; i < outputs.output_count; ++i) {
        ovrtx_render_var_output_handle_t ldr_handle = find_product_output(outputs.outputs[i], "LdrColor");
        if (ldr_handle != OVRTX_INVALID_HANDLE) {
            ovrtx_render_var_output_t ldr_output = {};
            ovrtx_result_t map_result = ovrtx_map_render_var_output(renderer, ldr_handle, &map_desc,
                                                                   ovrtx_timeout_infinite, &ldr_output);
            if (map_result.status == OVRTX_API_SUCCESS) {
                DLTensor const& t = *ldr_output.tensors[0].dl;
                std::string name = std::string("SensorConfig.") + product_names[i];
                save_ldr_png(name.c_str(), t.data,
                             static_cast<int>(t.shape[1]),
                             static_cast<int>(t.shape[0]));
                ovrtx_unmap_render_var_output(renderer, ldr_output.map_handle, no_sync);
            }
        }
    }

    ovrtx_destroy_results(renderer, step_handle);
    ovrtx_destroy_renderer(renderer);
}

TEST(SensorConfiguration, AddRenderConfigLayer) {
    TestConfig tc("SensorConfiguration.AddRenderConfigLayer");
    ovrtx_renderer_t* renderer = nullptr;
    ovrtx_result_t result = ovrtx_create_renderer(&tc.config, &renderer);
    ASSERT_API_SUCCESS(result.status);

    // [snippet:doc-add-render-config-layer-c]
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
    result = ovrtx_wait_op(renderer, enqueue_result.op_index,
                           ovrtx_timeout_infinite, &wait_result);
    ASSERT_API_SUCCESS(result.status);
    ASSERT_NO_OP_ERRORS(wait_result);
    // [/snippet:doc-add-render-config-layer-c]

    // Step to verify it works
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

    ovrtx_render_product_set_outputs_t outputs{};
    result = ovrtx_fetch_results(renderer, step_handle, ovrtx_timeout_infinite, &outputs);
    ASSERT_API_SUCCESS(result.status);
    EXPECT_GE(outputs.output_count, 1u);

    // Save LdrColor
    ovrtx_render_var_output_handle_t ldr_handle = find_output(outputs, "LdrColor");
    if (ldr_handle != OVRTX_INVALID_HANDLE) {
        ovrtx_map_output_description_t md = {};
        md.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;
        ovrtx_render_var_output_t ldr_output = {};
        ovrtx_result_t mr = ovrtx_map_render_var_output(renderer, ldr_handle, &md,
                                                       ovrtx_timeout_infinite, &ldr_output);
        if (mr.status == OVRTX_API_SUCCESS) {
            DLTensor const& t = *ldr_output.tensors[0].dl;
            save_ldr_png("SensorConfig.AddRenderConfigLayer",
                         t.data,
                         static_cast<int>(t.shape[1]),
                         static_cast<int>(t.shape[0]));
            ovrtx_cuda_sync_t no_sync = {};
            ovrtx_unmap_render_var_output(renderer, ldr_output.map_handle, no_sync);
        }
    }

    ovrtx_destroy_results(renderer, step_handle);
    ovrtx_destroy_renderer(renderer);
}
