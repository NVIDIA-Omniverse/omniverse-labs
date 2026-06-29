// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <volk.h>

// Wrapper around VkCommandBuffer providing command recording methods
class CommandBuffer {
public:
    explicit CommandBuffer(VkCommandBuffer cmd) : _cmd(cmd) {}
    
    // Get the underlying Vulkan handle
    auto handle() const -> VkCommandBuffer { return _cmd; }
    
    // Reset the command buffer for reuse
    auto reset() -> void;
    
    // Begin recording commands
    auto begin() -> void;
    
    // End recording commands
    auto end() -> void;

    // Dynamic rendering
    auto begin_rendering(VkRenderingInfo const& rendering_info) -> void;
    auto end_rendering() -> void;
    
    // Query pool operations
    auto reset_query_pool(VkQueryPool query_pool, uint32_t first_query, uint32_t query_count) -> void;
    auto write_timestamp(VkPipelineStageFlagBits stage, VkQueryPool query_pool, uint32_t query) -> void;
    
    // Synchronization (Vulkan 1.3)
    auto image_memory_barrier(VkImage image, VkImageAspectFlags aspect_mask,
                              VkImageLayout old_layout, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                              VkImageLayout new_layout, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                              uint32_t src_queue_family = VK_QUEUE_FAMILY_IGNORED,
                              uint32_t dst_queue_family = VK_QUEUE_FAMILY_IGNORED) -> void;
    
    // Dynamic state - viewport/scissor
    auto set_viewport(float x, float y, float width, float height, float min_depth = 0.0f, float max_depth = 1.0f) -> void;
    auto set_scissor(int32_t x, int32_t y, uint32_t width, uint32_t height) -> void;
    
    // Dynamic state - rasterization
    auto set_rasterizer_discard_enable(bool enable) -> void;
    auto set_polygon_mode(VkPolygonMode polygon_mode) -> void;
    auto set_cull_mode(VkCullModeFlags cull_mode) -> void;
    auto set_front_face(VkFrontFace front_face) -> void;
    auto set_depth_bias_enable(bool enable) -> void;
    
    // Dynamic state - primitive topology
    auto set_primitive_topology(VkPrimitiveTopology topology) -> void;
    auto set_primitive_restart_enable(bool enable) -> void;
    
    // Dynamic state - depth/stencil
    auto set_depth_test_enable(bool enable) -> void;
    auto set_depth_write_enable(bool enable) -> void;
    auto set_depth_bounds_test_enable(bool enable) -> void;
    auto set_stencil_test_enable(bool enable) -> void;
    
    // Dynamic state - multisample
    auto set_rasterization_samples(VkSampleCountFlagBits samples) -> void;
    auto set_sample_mask(VkSampleCountFlagBits samples, uint32_t mask) -> void;
    auto set_alpha_to_coverage_enable(bool enable) -> void;
    
    // Dynamic state - color blend
    auto set_color_blend_enable(uint32_t attachment, bool enable) -> void;
    auto set_color_write_mask(uint32_t attachment, VkColorComponentFlags mask) -> void;
    
    // Dynamic state - vertex input
    auto set_vertex_input_empty() -> void;
    
    // Descriptor sets
    auto bind_descriptor_sets(VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                              uint32_t first_set, uint32_t set_count, VkDescriptorSet const* sets) -> void;
    
    // Push constants
    auto push_constants(VkPipelineLayout layout, VkShaderStageFlags stage_flags,
                        uint32_t offset, uint32_t size, void const* data) -> void;
    
    // Draw commands
    auto draw(uint32_t vertex_count, uint32_t instance_count = 1, 
              uint32_t first_vertex = 0, uint32_t first_instance = 0) -> void;
    
private:
    VkCommandBuffer _cmd;
};
