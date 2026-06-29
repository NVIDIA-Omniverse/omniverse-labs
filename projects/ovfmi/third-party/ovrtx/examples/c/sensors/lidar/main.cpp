// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <ovrtx/ovrtx.h>
#include <ovrtx/ovrtx_config.h>
#include <ovrtx/ovrtx_types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>

namespace {

constexpr char kRenderProductPath[] = "/World/Render/Products/LidarProduct";
constexpr int kWarmupStepCount = 3;
constexpr double kStepDeltaTimeSeconds = 0.1;

#ifdef LIDAR_EXAMPLE_USDA_PATH
constexpr char kDefaultSceneFilePath[] = LIDAR_EXAMPLE_USDA_PATH;
#else
constexpr char kDefaultSceneFilePath[] = "lidar_example.usda";
#endif

// [snippet:ovx-string-view-helper]
ovx_string_t to_ovx_string(std::string_view value)
{
    return { value.data(), value.size() };
}
// [/snippet:ovx-string-view-helper]

// [snippet:string-helper]
// ovrtx strings carry pointer + length, so compare without requiring a
// null-terminated value from the API.
bool string_equals(ovx_string_t value, char const* expected)
{
    const size_t expected_length = std::strlen(expected);
    return value.ptr && value.length == expected_length &&
           std::strncmp(value.ptr, expected, expected_length) == 0;
}
// [/snippet:string-helper]

// [snippet:check-error-helper]
// Print the last ovrtx error for any API call that does not report success.
template <typename ResultT>
bool check_api_result(ResultT const& result, std::string_view operation)
{
    if (result.status == OVRTX_API_SUCCESS) {
        return true;
    }

    if (result.status == OVRTX_API_TIMEOUT) {
        std::cerr << "ovrtx " << operation << " timed out\n";
        return false;
    }

    ovx_string_t error = ovrtx_get_last_error();
    if (error.ptr && error.length > 0) {
        std::cerr << "ovrtx " << operation << " failed: "
                  << std::string_view(error.ptr, error.length) << "\n";
    } else {
        std::cerr << "ovrtx " << operation << " failed\n";
    }
    return false;
}
// [/snippet:check-error-helper]

// [snippet:wait-op-helper]
// ovrtx work is asynchronous. Wait for the enqueued operation and print any
// per-operation errors reported by the renderer.
bool wait_for_success(ovrtx_renderer_t* renderer,
                      ovrtx_op_id_t op_id,
                      std::string_view operation)
{
    ovrtx_op_wait_result_t wait_result = {};

    // Wait for the enqueue operation itself to complete.
    ovrtx_result_t result =
        ovrtx_wait_op(renderer, op_id, ovrtx_timeout_infinite, &wait_result);
    if (!check_api_result(result, operation)) {
        return false;
    }

    if (wait_result.num_error_ops == 0) {
        return true;
    }

    // A completed wait can still report failed child operations. Print their
    // messages so the sample is useful when a USD load or render step fails.
    for (size_t i = 0; i < wait_result.num_error_ops; ++i) {
        ovx_string_t msg = ovrtx_get_last_op_error(wait_result.error_op_ids[i]);
        if (msg.ptr && msg.length > 0) {
            std::cerr << operation << ": op " << wait_result.error_op_ids[i]
                      << " failed: " << std::string_view(msg.ptr, msg.length)
                      << "\n";
        } else {
            std::cerr << operation << ": op " << wait_result.error_op_ids[i]
                      << " failed\n";
        }
    }
    return false;
}
// [/snippet:wait-op-helper]

// [snippet:find-render-var-output-helper]
// Step results are organized by render product, frame, and render variable.
// This example has one render product and one PointCloud render variable.
ovrtx_render_var_output_handle_t find_render_var_output(
    ovrtx_render_product_set_outputs_t const& outputs,
    char const* render_var_name)
{
    for (size_t p = 0; p < outputs.output_count; ++p) {
        ovrtx_render_product_output_t const& product = outputs.outputs[p];

        // A render product can produce multiple frames; each frame owns the
        // render variable outputs produced for that frame.
        for (size_t f = 0; f < product.output_frame_count; ++f) {
            ovrtx_render_product_frame_output_t const& frame =
                product.output_frames[f];
            for (size_t v = 0; v < frame.render_var_count; ++v) {
                ovrtx_render_product_render_var_output_t const& var =
                    frame.output_render_vars[v];
                if (string_equals(var.render_var_name, render_var_name)) {
                    return var.output_handle;
                }
            }
        }
    }
    return OVRTX_INVALID_HANDLE;
}
// [/snippet:find-render-var-output-helper]

// [snippet:find-tensor-helper]
// A composite render variable exposes named tensors. Find the channel tensor
// requested in lidar_example.usda.
ovrtx_render_var_tensor_t const* find_tensor(ovrtx_render_var_output_t const& output,
                                             char const* name)
{
    // Tensor names are ovrtx strings, so compare by length and content.
    for (size_t i = 0; i < output.num_tensors; ++i) {
        ovrtx_render_var_tensor_t const& tensor = output.tensors[i];
        if (tensor.name && string_equals(*tensor.name, name)) {
            return &tensor;
        }
    }
    return nullptr;
}
// [/snippet:find-tensor-helper]

// [snippet:cpu-tensor-helper]
// Return host-readable data for a named channel in the CPU-mapped composite
// output. The template type matches the DLTensor dtype requested by the USD.
template <typename T>
T const* cpu_tensor_data(ovrtx_render_var_output_t const& output,
                         char const* name)
{
    ovrtx_render_var_tensor_t const* tensor = find_tensor(output, name);

    // After CPU mapping, each requested channel should expose a CPU DLTensor.
    if (!tensor || !tensor->dl || tensor->dl->device.device_type != kDLCPU ||
        !tensor->dl->data) {
        std::cerr << "Lidar PointCloud tensor '" << name
                  << "' is not available on CPU\n";
        return nullptr;
    }
    return static_cast<T const*>(tensor->dl->data);
}
// [/snippet:cpu-tensor-helper]

// [snippet:read-lidar-pointcloud]
// Read one mapped lidar PointCloud. Counts is the valid point count; Intensity
// and TimeOffsetNs are per-point channels over that valid range.
bool print_pointcloud_summary(ovrtx_render_var_output_t const& output)
{
    // The USD also requests Coordinates; this example only needs scalar and
    // time channels for the printed statistics.
    int32_t const* counts_data = cpu_tensor_data<int32_t>(output, "Counts");
    float const* intensity_data = cpu_tensor_data<float>(output, "Intensity");
    int32_t const* time_offset_data =
        cpu_tensor_data<int32_t>(output, "TimeOffsetNs");

    if (!counts_data || !intensity_data || !time_offset_data) {
        return false;
    }

    // Counts is a scalar tensor containing the number of valid point entries.
    int32_t const valid_point_count = counts_data[0];
    if (valid_point_count <= 0) {
        std::cerr << "Expected at least one valid lidar point, got "
                  << valid_point_count << "\n";
        return false;
    }

    double intensity_sum = 0.0;
    int32_t max_time_offset_ns = time_offset_data[0];

    // Only the first Counts entries are valid in each per-point channel.
    for (int32_t i = 0; i < valid_point_count; ++i) {
        const float intensity = intensity_data[i];
        if (!std::isfinite(intensity)) {
            std::cerr << "Intensity tensor contained a non-finite value\n";
            return false;
        }
        intensity_sum += intensity;
        max_time_offset_ns = std::max(max_time_offset_ns, time_offset_data[i]);
    }

    const double mean_intensity =
        intensity_sum / static_cast<double>(valid_point_count);
    std::cout << "valid points=" << valid_point_count
              << ", mean intensity=" << mean_intensity
              << ", max point time offset to point cloud frame start=" << max_time_offset_ns << " ns\n";
    return true;
}
// [/snippet:read-lidar-pointcloud]

// [snippet:destroy-results-helper]
// Result handles own step output memory and must be destroyed after use.
bool destroy_step_results(ovrtx_renderer_t* renderer,
                          ovrtx_step_result_handle_t* step_handle)
{
    if (*step_handle == OVRTX_INVALID_HANDLE) {
        return true;
    }

    ovrtx_result_t result = ovrtx_destroy_results(renderer, *step_handle);

    // Clear the handle so repeated cleanup attempts are harmless.
    *step_handle = OVRTX_INVALID_HANDLE;
    return check_api_result(result, "destroy_results");
}
// [/snippet:destroy-results-helper]

// [snippet:cleanup-helper]
// Destroy the renderer on every exit path without obscuring the main flow.
int finish(ovrtx_renderer_t* renderer, int exit_code)
{
    if (renderer) {
        ovrtx_result_t result = ovrtx_destroy_renderer(renderer);
        if (!check_api_result(result, "destroy_renderer")) {
            return 1;
        }
    }
    return exit_code;
}
// [/snippet:cleanup-helper]

} // namespace

int main(int argc, char* argv[])
{
    ovrtx_renderer_t* renderer = nullptr;
    ovrtx_enqueue_result_t enqueue_result = {};
    ovrtx_result_t result = {};
    ovx_string_t render_product_path = {};
    ovrtx_render_product_set_t render_products = {};

    // Use the CMake-configured scene path by default; argv[1] lets copied
    // builds run the same binary against another USDA file.
    char const* scene_file_path =
        (argc > 1) ? argv[1] : kDefaultSceneFilePath;

    // [snippet:create-renderer]
    // Enable motion BVH: required by the lidar sensor pipeline.
    ovrtx_config_entry_t config_entries[] = {
        ovrtx_config_entry_enable_motion_bvh(true),
    };
    ovrtx_config_t config = { config_entries, 1 };

    // Create the renderer before opening the USDA scene.
    result = ovrtx_create_renderer(&config, &renderer);
    if (!check_api_result(result, "create_renderer")) {
        return finish(renderer, 1);
    }
    // [/snippet:create-renderer]

    // [snippet:load-lidar-scene]
    std::cout << "Loading lidar scene from " << scene_file_path << "...\n";
    enqueue_result =
        ovrtx_open_usd_from_file(renderer, to_ovx_string(scene_file_path));
    if (!check_api_result(enqueue_result, "open_usd_from_file") ||
        !wait_for_success(renderer, enqueue_result.op_index, "open_usd_from_file")) {
        return finish(renderer, 1);
    }
    // [/snippet:load-lidar-scene]

    // [snippet:configure-render-product]
    render_product_path = { kRenderProductPath, std::strlen(kRenderProductPath) };
    render_products.render_products = &render_product_path;
    render_products.num_render_products = 1;
    // [/snippet:configure-render-product]

    // [snippet:warm-up-lidar]
    // Render a few warmup frames without reading outputs.
    for (int i = 0; i < kWarmupStepCount; ++i) {
        ovrtx_step_result_handle_t warmup_handle = OVRTX_INVALID_HANDLE;

        enqueue_result =
            ovrtx_step(renderer,
                       render_products,
                       kStepDeltaTimeSeconds,
                       &warmup_handle);
        if (!check_api_result(enqueue_result, "step") ||
            !wait_for_success(renderer, enqueue_result.op_index, "step")) {
            destroy_step_results(renderer, &warmup_handle);
            return finish(renderer, 1);
        }
        if (!destroy_step_results(renderer, &warmup_handle)) {
            return finish(renderer, 1);
        }
    }
    // [/snippet:warm-up-lidar]

    // [snippet:step-lidar-pointcloud]
    ovrtx_step_result_handle_t step_handle = OVRTX_INVALID_HANDLE;

    // Render one lidar frame and keep the result handle for fetching.
    enqueue_result =
        ovrtx_step(renderer,
                   render_products,
                   kStepDeltaTimeSeconds,
                   &step_handle);
    if (!check_api_result(enqueue_result, "step") ||
        !wait_for_success(renderer, enqueue_result.op_index, "step")) {
        destroy_step_results(renderer, &step_handle);
        return finish(renderer, 1);
    }

    // Fetch the completed step and locate the PointCloud render variable.
    ovrtx_render_product_set_outputs_t outputs = {};
    result = ovrtx_fetch_results(
        renderer, step_handle, ovrtx_timeout_infinite, &outputs);
    if (!check_api_result(result, "fetch_results")) {
        destroy_step_results(renderer, &step_handle);
        return finish(renderer, 1);
    }

    ovrtx_render_var_output_handle_t pointcloud_handle =
        find_render_var_output(outputs, "PointCloud");
    if (pointcloud_handle == OVRTX_INVALID_HANDLE) {
        std::cerr << "PointCloud render var output not found\n";
        destroy_step_results(renderer, &step_handle);
        return finish(renderer, 1);
    }

    // Map the composite PointCloud render variable to CPU. The mapped output
    // exposes one DLTensor per channel requested in the USD.
    ovrtx_map_output_description_t map_desc = {};
    map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;
    ovrtx_render_var_output_t pointcloud_output = {};
    result = ovrtx_map_render_var_output(renderer,
                                         pointcloud_handle,
                                         &map_desc,
                                         ovrtx_timeout_infinite,
                                         &pointcloud_output);
    if (!check_api_result(result, "map_render_var_output")) {
        destroy_step_results(renderer, &step_handle);
        return finish(renderer, 1);
    }

    const bool printed_summary = print_pointcloud_summary(pointcloud_output);

    // Always unmap after reading the CPU DLTensor views.
    ovrtx_cuda_sync_t no_sync = {};
    result = ovrtx_unmap_render_var_output(
        renderer, pointcloud_output.map_handle, no_sync);
    if (!check_api_result(result, "unmap_render_var_output") ||
        !printed_summary) {
        destroy_step_results(renderer, &step_handle);
        return finish(renderer, 1);
    }

    if (!destroy_step_results(renderer, &step_handle)) {
        return finish(renderer, 1);
    }
    // [/snippet:step-lidar-pointcloud]

    return finish(renderer, 0);
}
