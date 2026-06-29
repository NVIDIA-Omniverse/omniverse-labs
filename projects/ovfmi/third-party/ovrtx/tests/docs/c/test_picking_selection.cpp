// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Tests for viewport picking and selection outline drawing.

#include <gtest/gtest.h>
#include "helpers.h"

#include <ovrtx/ovrtx_attributes.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <string_view>
#include <vector>

static bool ovx_string_equals(ovx_string_t value, std::string_view expected) {
    return value.ptr && value.length == expected.size() &&
           std::string_view(value.ptr, value.length) == expected;
}

static DLTensor const* find_tensor(ovrtx_render_var_output_t const& output,
                                   std::string_view name) {
    for (size_t i = 0; i < output.num_tensors; ++i) {
        ovrtx_render_var_tensor_t const& tensor = output.tensors[i];
        if (tensor.name && ovx_string_equals(*tensor.name, name)) {
            return tensor.dl;
        }
    }
    return nullptr;
}

static DLTensor const* find_param(ovrtx_render_var_output_t const& output,
                                  std::string_view name) {
    for (size_t i = 0; i < output.num_params; ++i) {
        ovrtx_render_var_param_t const& param = output.params[i];
        if (ovx_string_equals(param.name, name)) {
            return &param.dl;
        }
    }
    return nullptr;
}

static constexpr ovrtx_selection_group_style_t DEFAULT_SELECTION_STYLE = {
    {1.0f, 0.6f, 0.0f, 1.0f},
    {0.0f, 0.0f, 0.0f, 0.0f},
};
static constexpr int32_t kRenderWidth = 640;
static constexpr int32_t kRenderHeight = 320;
static constexpr int32_t kCenterX = kRenderWidth / 2;
static constexpr int32_t kCenterY = kRenderHeight / 2;
static constexpr int32_t kMarqueeRight = kRenderWidth * 150 / 256;

class PickingSelectionTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // [snippet:doc-create-selection-outline-renderer-c]
        std::string log_path = (get_output_dir() / "PickingSelectionTest-ovrtx.log").string();
        ovx_string_t log_path_view = {log_path.c_str(), log_path.size()};

        ovrtx_config_entry_t entries[] = {
            ovrtx_config_entry_log_file_path(log_path_view),
            ovrtx_config_entry_selection_outline_enabled(true),
        };
        ovrtx_config_t config = {entries, 2};

        ovrtx_result_t result = ovrtx_create_renderer(&config, &renderer_);
        ASSERT_API_SUCCESS(result.status);
        // [/snippet:doc-create-selection-outline-renderer-c]
    }

    static void TearDownTestSuite() {
        if (renderer_) {
            ovrtx_destroy_renderer(renderer_);
            renderer_ = nullptr;
        }
    }

    static void load_scene() {
        ovrtx_enqueue_result_t eq = ovrtx_reset_stage(renderer_);
        ASSERT_API_SUCCESS(eq.status);
        docs_wait_no_errors(renderer_, eq.op_index);

        std::string scene = get_docs_test_data_dir() + "/ovrtx-test-picking-selection.usda";
        eq = ovrtx_open_usd_from_file(renderer_, {scene.c_str(), scene.size()});
        ASSERT_API_SUCCESS(eq.status);
        docs_wait_no_errors(renderer_, eq.op_index);

        eq = ovrtx_reset(renderer_, 0.0);
        ASSERT_API_SUCCESS(eq.status);
        docs_wait_no_errors(renderer_, eq.op_index);

        for (int i = 0; i < 2; ++i) {
            ovrtx_step_result_handle_t step_handle = step_once();
            ovrtx_destroy_results(renderer_, step_handle);
        }
    }

    static ovrtx_render_product_set_t render_products() {
        static ovx_string_t rp_path = ovx_str("/Render/Camera");
        ovrtx_render_product_set_t products{};
        products.render_products = &rp_path;
        products.num_render_products = 1;
        return products;
    }

    static ovrtx_step_result_handle_t step_once() {
        ovrtx_step_result_handle_t step_handle = 0;
        ovrtx_render_product_set_t products = render_products();
        ovrtx_enqueue_result_t eq = ovrtx_step(renderer_, products, 1.0 / 60.0, &step_handle);
        EXPECT_API_SUCCESS(eq.status);
        if (eq.status == OVRTX_API_SUCCESS) {
            docs_wait_no_errors(renderer_, eq.op_index);
        }
        return step_handle;
    }

    static void enqueue_pick_rect(int32_t left, int32_t top, int32_t right, int32_t bottom) {
        ovx_string_t rp_path = ovx_str("/Render/Camera");

        // [snippet:doc-enqueue-pick-query-c]
        ovrtx_pick_query_desc_t pick_desc = {};
        pick_desc.render_product_path = rp_path;
        pick_desc.left = left;
        pick_desc.top = top;
        pick_desc.right = right;
        pick_desc.bottom = bottom;
        pick_desc.flags = 0;

        ovrtx_enqueue_result_t enqueue_result =
            ovrtx_enqueue_pick_query(renderer_, &pick_desc);
        ASSERT_API_SUCCESS(enqueue_result.status);
        docs_wait_no_errors(renderer_, enqueue_result.op_index);
        // [/snippet:doc-enqueue-pick-query-c]
    }

    static std::set<std::string> collect_pick_paths(ovrtx_step_result_handle_t step_handle) {
        std::set<std::string> paths;

        ovrtx_render_product_set_outputs_t outputs{};
        ovrtx_result_t result =
            ovrtx_fetch_results(renderer_, step_handle, ovrtx_timeout_infinite, &outputs);
        EXPECT_API_SUCCESS(result.status);
        if (result.status != OVRTX_API_SUCCESS) {
            return paths;
        }

        ovrtx_render_var_output_handle_t pick_handle = find_output(outputs, OVRTX_RENDER_VAR_PICK_HIT);
        EXPECT_NE(pick_handle, OVRTX_INVALID_HANDLE);
        if (pick_handle == OVRTX_INVALID_HANDLE) {
            return paths;
        }

        ovrtx_map_output_description_t map_desc = {};
        map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;

        ovrtx_render_var_output_t pick_output = {};
        result = ovrtx_map_render_var_output(
            renderer_, pick_handle, &map_desc, ovrtx_timeout_infinite, &pick_output);
        EXPECT_API_SUCCESS(result.status);
        if (result.status != OVRTX_API_SUCCESS) {
            return paths;
        }

        // [snippet:doc-read-pick-hit-buffer-c]
        DLTensor const* magic_param = find_param(pick_output, "magic");
        DLTensor const* version_param = find_param(pick_output, "version");
        DLTensor const* hit_count_param = find_param(pick_output, "hitCount");
        DLTensor const* prim_path_tensor = find_tensor(pick_output, "primPath");
        DLTensor const* world_position_tensor = find_tensor(pick_output, "worldPositionM");
        DLTensor const* world_normal_tensor = find_tensor(pick_output, "worldNormal");

        EXPECT_NE(magic_param, nullptr);
        EXPECT_NE(version_param, nullptr);
        EXPECT_NE(hit_count_param, nullptr);
        EXPECT_NE(prim_path_tensor, nullptr);
        EXPECT_NE(world_position_tensor, nullptr);
        EXPECT_NE(world_normal_tensor, nullptr);
        if (!magic_param || !version_param || !hit_count_param ||
            !prim_path_tensor || !world_position_tensor || !world_normal_tensor) {
            ovrtx_cuda_sync_t no_sync = {};
            EXPECT_API_SUCCESS(ovrtx_unmap_render_var_output(renderer_, pick_output.map_handle, no_sync).status);
            return paths;
        }
        EXPECT_NE(magic_param->data, nullptr);
        EXPECT_NE(version_param->data, nullptr);
        EXPECT_NE(hit_count_param->data, nullptr);
        EXPECT_NE(prim_path_tensor->data, nullptr);
        EXPECT_NE(prim_path_tensor->shape, nullptr);
        EXPECT_NE(world_position_tensor->data, nullptr);
        EXPECT_NE(world_position_tensor->shape, nullptr);
        EXPECT_NE(world_normal_tensor->data, nullptr);
        EXPECT_NE(world_normal_tensor->shape, nullptr);
        if (!magic_param->data || !version_param->data || !hit_count_param->data ||
            !prim_path_tensor->data || !prim_path_tensor->shape ||
            !world_position_tensor->data || !world_position_tensor->shape ||
            !world_normal_tensor->data || !world_normal_tensor->shape) {
            ovrtx_cuda_sync_t no_sync = {};
            EXPECT_API_SUCCESS(ovrtx_unmap_render_var_output(renderer_, pick_output.map_handle, no_sync).status);
            return paths;
        }

        uint32_t magic = *static_cast<uint32_t const*>(magic_param->data);
        uint32_t version = *static_cast<uint32_t const*>(version_param->data);
        uint32_t hit_count = *static_cast<uint32_t const*>(hit_count_param->data);
        EXPECT_EQ(magic, OVRTX_PICK_HIT_MAGIC);
        EXPECT_EQ(version, OVRTX_PICK_HIT_VERSION);
        EXPECT_EQ(prim_path_tensor->ndim, 1);
        EXPECT_GE(prim_path_tensor->shape[0], static_cast<int64_t>(hit_count));
        EXPECT_EQ(world_position_tensor->ndim, 2);
        EXPECT_GE(world_position_tensor->shape[0], static_cast<int64_t>(hit_count));
        EXPECT_GE(world_position_tensor->shape[1], 3);
        EXPECT_EQ(world_normal_tensor->ndim, 2);
        EXPECT_GE(world_normal_tensor->shape[0], static_cast<int64_t>(hit_count));
        EXPECT_GE(world_normal_tensor->shape[1], 3);
        if (prim_path_tensor->ndim != 1 ||
            prim_path_tensor->shape[0] < static_cast<int64_t>(hit_count) ||
            world_position_tensor->ndim != 2 ||
            world_position_tensor->shape[0] < static_cast<int64_t>(hit_count) ||
            world_position_tensor->shape[1] < 3 ||
            world_normal_tensor->ndim != 2 ||
            world_normal_tensor->shape[0] < static_cast<int64_t>(hit_count) ||
            world_normal_tensor->shape[1] < 3) {
            ovrtx_cuda_sync_t no_sync = {};
            EXPECT_API_SUCCESS(ovrtx_unmap_render_var_output(renderer_, pick_output.map_handle, no_sync).status);
            return paths;
        }

        const auto* prim_paths =
            static_cast<const ovx_primpath_t*>(prim_path_tensor->data);
        const auto* world_positions =
            static_cast<const double*>(world_position_tensor->data);
        const auto* world_normals =
            static_cast<const float*>(world_normal_tensor->data);
        std::vector<ovx_primpath_t> prim_path_ids;
        for (uint32_t i = 0; i < hit_count; ++i) {
            EXPECT_NE(prim_paths[i], 0u);
            EXPECT_TRUE(std::isfinite(world_positions[i * 3 + 0]));
            EXPECT_TRUE(std::isfinite(world_positions[i * 3 + 1]));
            EXPECT_TRUE(std::isfinite(world_positions[i * 3 + 2]));
            EXPECT_TRUE(std::isfinite(world_normals[i * 3 + 0]));
            EXPECT_TRUE(std::isfinite(world_normals[i * 3 + 1]));
            EXPECT_TRUE(std::isfinite(world_normals[i * 3 + 2]));
            prim_path_ids.push_back(prim_paths[i]);
        }
        // [/snippet:doc-read-pick-hit-buffer-c]

        // [snippet:doc-resolve-picked-prim-paths-c]
        path_dictionary_instance_t path_dictionary = {};
        ovrtx_result_t path_dictionary_result =
            ovrtx_get_path_dictionary(renderer_, &path_dictionary);
        EXPECT_API_SUCCESS(path_dictionary_result.status);

        if (path_dictionary_result.status == OVRTX_API_SUCCESS) {
            for (ovx_primpath_t prim_path : prim_path_ids) {
                std::string path = docs_resolve_primpath(&path_dictionary, prim_path);
                if (!path.empty()) {
                    paths.insert(path);
                }
            }
        }
        // [/snippet:doc-resolve-picked-prim-paths-c]

        ovrtx_cuda_sync_t no_sync = {};
        EXPECT_API_SUCCESS(ovrtx_unmap_render_var_output(renderer_, pick_output.map_handle, no_sync).status);
        return paths;
    }

    static std::vector<uint8_t> render_ldr_pixels() {
        ovrtx_step_result_handle_t step_handle = step_once();

        ovrtx_render_product_set_outputs_t outputs{};
        ovrtx_result_t result =
            ovrtx_fetch_results(renderer_, step_handle, ovrtx_timeout_infinite, &outputs);
        EXPECT_API_SUCCESS(result.status);
        if (result.status != OVRTX_API_SUCCESS) {
            ovrtx_destroy_results(renderer_, step_handle);
            return {};
        }

        ovrtx_render_var_output_handle_t ldr_handle = find_output(outputs, "LdrColor");
        EXPECT_NE(ldr_handle, OVRTX_INVALID_HANDLE);
        if (ldr_handle == OVRTX_INVALID_HANDLE) {
            ovrtx_destroy_results(renderer_, step_handle);
            return {};
        }

        ovrtx_map_output_description_t map_desc = {};
        map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;

        ovrtx_render_var_output_t ldr_output = {};
        result = ovrtx_map_render_var_output(
            renderer_, ldr_handle, &map_desc, ovrtx_timeout_infinite, &ldr_output);
        EXPECT_API_SUCCESS(result.status);
        if (result.status != OVRTX_API_SUCCESS) {
            ovrtx_destroy_results(renderer_, step_handle);
            return {};
        }

        EXPECT_GT(ldr_output.num_tensors, 0u);
        if (ldr_output.num_tensors == 0) {
            ovrtx_cuda_sync_t no_sync = {};
            EXPECT_API_SUCCESS(ovrtx_unmap_render_var_output(renderer_, ldr_output.map_handle, no_sync).status);
            ovrtx_destroy_results(renderer_, step_handle);
            return {};
        }
        DLTensor const& tensor = *ldr_output.tensors[0].dl;
        EXPECT_EQ(tensor.shape[0], kRenderHeight);
        EXPECT_EQ(tensor.shape[1], kRenderWidth);
        EXPECT_EQ(tensor.shape[2], 4);

        size_t const byte_count =
            static_cast<size_t>(tensor.shape[0]) *
            static_cast<size_t>(tensor.shape[1]) *
            static_cast<size_t>(tensor.shape[2]);
        auto const* bytes = static_cast<uint8_t const*>(tensor.data);
        std::vector<uint8_t> pixels(bytes, bytes + byte_count);

        ovrtx_cuda_sync_t no_sync = {};
        EXPECT_API_SUCCESS(ovrtx_unmap_render_var_output(renderer_, ldr_output.map_handle, no_sync).status);
        ovrtx_destroy_results(renderer_, step_handle);
        return pixels;
    }

    static size_t count_changed_pixels(std::vector<uint8_t> const& a,
                                       std::vector<uint8_t> const& b,
                                       int threshold = 8) {
        EXPECT_EQ(a.size(), b.size());
        if (a.size() != b.size() || a.size() % 4 != 0) {
            return 0;
        }

        size_t changed = 0;
        for (size_t i = 0; i < a.size(); i += 4) {
            bool pixel_changed = false;
            for (size_t channel = 0; channel < 4; ++channel) {
                int delta = std::abs(static_cast<int>(a[i + channel]) - static_cast<int>(b[i + channel]));
                pixel_changed = pixel_changed || delta > threshold;
            }
            if (pixel_changed) {
                ++changed;
            }
        }
        return changed;
    }

    static std::vector<uint8_t> make_delta_image(std::vector<uint8_t> const& a,
                                                 std::vector<uint8_t> const& b) {
        EXPECT_EQ(a.size(), b.size());
        if (a.size() != b.size()) {
            return {};
        }

        std::vector<uint8_t> diff(a.size(), 0);
        for (size_t i = 0; i < a.size(); i += 4) {
            for (size_t channel = 0; channel < 3; ++channel) {
                int delta = std::abs(static_cast<int>(a[i + channel]) - static_cast<int>(b[i + channel]));
                diff[i + channel] = static_cast<uint8_t>(std::min(delta * 8, 255));
            }
            diff[i + 3] = 255;
        }
        return diff;
    }

    static void save_selection_outline_images(std::vector<uint8_t> const& baseline,
                                              std::vector<uint8_t> const& selected,
                                              std::vector<uint8_t> const& cleared) {
        save_ldr_png(
            "PickingSelection.SelectionOutline.baseline",
            baseline.data(),
            kRenderWidth,
            kRenderHeight);
        save_ldr_png(
            "PickingSelection.SelectionOutline.selected",
            selected.data(),
            kRenderWidth,
            kRenderHeight);
        save_ldr_png(
            "PickingSelection.SelectionOutline.cleared",
            cleared.data(),
            kRenderWidth,
            kRenderHeight);

        std::vector<uint8_t> selected_vs_baseline = make_delta_image(selected, baseline);
        std::vector<uint8_t> selected_vs_cleared = make_delta_image(selected, cleared);
        if (!selected_vs_baseline.empty()) {
            save_ldr_png("PickingSelection.SelectionOutline.delta_selected_vs_baseline",
                         selected_vs_baseline.data(),
                         kRenderWidth,
                         kRenderHeight);
        }
        if (!selected_vs_cleared.empty()) {
            save_ldr_png("PickingSelection.SelectionOutline.delta_selected_vs_cleared",
                         selected_vs_cleared.data(),
                         kRenderWidth,
                         kRenderHeight);
        }
    }

    static void restore_default_selection_styles() {
        if (!renderer_) {
            return;
        }
        const uint8_t group_ids[] = {1u, 2u};
        const ovrtx_selection_group_style_t styles[] = {
            DEFAULT_SELECTION_STYLE,
            DEFAULT_SELECTION_STYLE,
        };
        ovrtx_enqueue_result_t enqueue_result =
            ovrtx_set_selection_group_styles(renderer_, group_ids, styles, 2);
        if (enqueue_result.status == OVRTX_API_SUCCESS) {
            docs_wait_no_errors(renderer_, enqueue_result.op_index);
        }
    }

    static ovrtx_renderer_t* renderer_;
};

ovrtx_renderer_t* PickingSelectionTest::renderer_ = nullptr;

class SelectionStyleTest : public PickingSelectionTest {
protected:
    static void SetUpTestSuite() {
        // [snippet:doc-create-styled-selection-renderer-c]
        std::string log_path = (get_output_dir() / "SelectionStyleTest-ovrtx.log").string();
        ovx_string_t log_path_view = {log_path.c_str(), log_path.size()};

        ovrtx_config_entry_t entries[] = {
            ovrtx_config_entry_log_file_path(log_path_view),
            ovrtx_config_entry_selection_outline_enabled(true),
            ovrtx_config_entry_selection_outline_width(8),
            ovrtx_config_entry_selection_fill_mode(OVRTX_SELECTION_FILL_MODE_GROUP_FILL_COLOR),
        };
        ovrtx_config_t config = {entries, 4};

        ovrtx_result_t result = ovrtx_create_renderer(&config, &renderer_);
        ASSERT_API_SUCCESS(result.status);
        // [/snippet:doc-create-styled-selection-renderer-c]
    }

    static void TearDownTestSuite() {
        restore_default_selection_styles();
        PickingSelectionTest::TearDownTestSuite();
    }
};

TEST_F(PickingSelectionTest, PickCenterPixel) {
    load_scene();

    enqueue_pick_rect(kCenterX, kCenterY, kCenterX + 1, kCenterY + 1);
    ovrtx_step_result_handle_t step_handle = step_once();

    std::set<std::string> picked = collect_pick_paths(step_handle);
    EXPECT_EQ(picked, std::set<std::string>({"/World/CenterCube"}));

    ovrtx_destroy_results(renderer_, step_handle);
}

TEST_F(PickingSelectionTest, MarqueePicksMultiplePrims) {
    load_scene();

    enqueue_pick_rect(0, 0, kMarqueeRight, kRenderHeight);
    ovrtx_step_result_handle_t step_handle = step_once();

    std::set<std::string> picked = collect_pick_paths(step_handle);
    EXPECT_TRUE(picked.count("/World/LeftCube") > 0);
    EXPECT_TRUE(picked.count("/World/CenterCube") > 0);
    EXPECT_TRUE(picked.count("/World/RightCube") == 0);

    ovrtx_destroy_results(renderer_, step_handle);
}

TEST_F(PickingSelectionTest, PickableFalseExcludesPrim) {
    load_scene();

    enqueue_pick_rect(kCenterX, kCenterY, kCenterX + 1, kCenterY + 1);
    ovrtx_step_result_handle_t step_handle = step_once();
    std::set<std::string> picked = collect_pick_paths(step_handle);
    EXPECT_EQ(picked, std::set<std::string>({"/World/CenterCube"}));
    ovrtx_destroy_results(renderer_, step_handle);

    // [snippet:doc-set-pickable-c]
    ovx_string_t unpickable_path = ovx_str("/World/CenterCube");
    bool pickable = false;

    ovrtx_enqueue_result_t enqueue_result =
        ovrtx_set_pickable(renderer_, &unpickable_path, 1, &pickable);
    ASSERT_API_SUCCESS(enqueue_result.status);
    docs_wait_no_errors(renderer_, enqueue_result.op_index);
    // [/snippet:doc-set-pickable-c]

    enqueue_pick_rect(0, 0, kMarqueeRight, kRenderHeight);
    step_handle = step_once();

    picked = collect_pick_paths(step_handle);
    EXPECT_TRUE(picked.count("/World/LeftCube") > 0);
    EXPECT_TRUE(picked.count("/World/CenterCube") == 0);

    ovrtx_destroy_results(renderer_, step_handle);
}

TEST_F(PickingSelectionTest, SelectionOutlineGroupRenders) {
    load_scene();

    std::vector<uint8_t> baseline_pixels = render_ldr_pixels();
    ASSERT_FALSE(baseline_pixels.empty());

    // [snippet:doc-set-selection-outline-group-c]
    ovx_string_t selected_path = ovx_str("/World/CenterCube");
    uint8_t outline_group = 1;

    ovrtx_enqueue_result_t enqueue_result =
        ovrtx_set_selection_outline_group(renderer_, &selected_path, 1, &outline_group);
    ASSERT_API_SUCCESS(enqueue_result.status);
    docs_wait_no_errors(renderer_, enqueue_result.op_index);
    // [/snippet:doc-set-selection-outline-group-c]

    std::vector<uint8_t> selected_pixels = render_ldr_pixels();
    ASSERT_FALSE(selected_pixels.empty());
    EXPECT_GT(count_changed_pixels(selected_pixels, baseline_pixels), 0u);

    // [snippet:doc-clear-selection-outline-group-c]
    outline_group = 0;

    enqueue_result =
        ovrtx_set_selection_outline_group(renderer_, &selected_path, 1, &outline_group);
    ASSERT_API_SUCCESS(enqueue_result.status);
    docs_wait_no_errors(renderer_, enqueue_result.op_index);
    // [/snippet:doc-clear-selection-outline-group-c]

    std::vector<uint8_t> cleared_pixels = render_ldr_pixels();
    ASSERT_FALSE(cleared_pixels.empty());
    save_selection_outline_images(baseline_pixels, selected_pixels, cleared_pixels);
    EXPECT_GT(count_changed_pixels(selected_pixels, cleared_pixels), 0u);
}

TEST_F(SelectionStyleTest, SelectionStyleGroupsControlOutlineAndFill) {
    load_scene();

    std::vector<uint8_t> baseline_pixels = render_ldr_pixels();
    ASSERT_FALSE(baseline_pixels.empty());

    // [snippet:doc-set-selection-group-styles-c]
    const uint8_t group_ids[] = {1u, 2u};
    const ovrtx_selection_group_style_t styles[] = {
        {{1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        {{0.0f, 0.0f, 1.0f, 1.0f}, {1.0f, 0.0f, 1.0f, 1.0f}},
    };

    ovrtx_enqueue_result_t enqueue_result =
        ovrtx_set_selection_group_styles(renderer_, group_ids, styles, 2);
    ASSERT_API_SUCCESS(enqueue_result.status);
    docs_wait_no_errors(renderer_, enqueue_result.op_index);
    // [/snippet:doc-set-selection-group-styles-c]

    // [snippet:doc-assign-selection-style-groups-c]
    ovx_string_t selected_paths[] = {
        ovx_str("/World/CenterCube"),
        ovx_str("/World/LeftCube"),
    };
    uint8_t outline_groups[] = {1u, 2u};

    enqueue_result =
        ovrtx_set_selection_outline_group(renderer_, selected_paths, 2, outline_groups);
    ASSERT_API_SUCCESS(enqueue_result.status);
    docs_wait_no_errors(renderer_, enqueue_result.op_index);
    // [/snippet:doc-assign-selection-style-groups-c]

    std::vector<uint8_t> styled_pixels = render_ldr_pixels();
    ASSERT_FALSE(styled_pixels.empty());
    EXPECT_GT(count_changed_pixels(styled_pixels, baseline_pixels), 0u);

    const ovrtx_selection_group_style_t fill_recolor_styles[] = {
        {{1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
        {{0.0f, 0.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
    };
    enqueue_result =
        ovrtx_set_selection_group_styles(renderer_, group_ids, fill_recolor_styles, 2);
    ASSERT_API_SUCCESS(enqueue_result.status);
    docs_wait_no_errors(renderer_, enqueue_result.op_index);

    std::vector<uint8_t> fill_recolored_pixels = render_ldr_pixels();
    ASSERT_FALSE(fill_recolored_pixels.empty());
    EXPECT_GT(count_changed_pixels(fill_recolored_pixels, styled_pixels), 0u);

    uint8_t swapped_outline_groups[] = {2u, 1u};
    enqueue_result =
        ovrtx_set_selection_outline_group(renderer_, selected_paths, 2, swapped_outline_groups);
    ASSERT_API_SUCCESS(enqueue_result.status);
    docs_wait_no_errors(renderer_, enqueue_result.op_index);

    std::vector<uint8_t> swapped_pixels = render_ldr_pixels();
    ASSERT_FALSE(swapped_pixels.empty());
    EXPECT_GT(count_changed_pixels(swapped_pixels, fill_recolored_pixels), 0u);
}
