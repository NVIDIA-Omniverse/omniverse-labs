// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "command_buffer.hpp"

auto CommandBuffer::reset() -> void {
    vkResetCommandBuffer(_cmd, 0);
}

auto CommandBuffer::begin() -> void {
    vkResetCommandBuffer(_cmd, 0);
    
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(_cmd, &begin_info);
}

auto CommandBuffer::end() -> void {
    vkEndCommandBuffer(_cmd);
}

auto CommandBuffer::begin_rendering(
    VkRenderingInfo const& rendering_info) -> void {
    vkCmdBeginRendering(_cmd, &rendering_info);
}

auto CommandBuffer::end_rendering() -> void {
    vkCmdEndRendering(_cmd);
}

auto CommandBuffer::reset_query_pool(VkQueryPool query_pool, uint32_t first_query, uint32_t query_count) -> void {
    vkCmdResetQueryPool(_cmd, query_pool, first_query, query_count);
}

auto CommandBuffer::write_timestamp(VkPipelineStageFlagBits stage, VkQueryPool query_pool, uint32_t query) -> void {
    vkCmdWriteTimestamp(_cmd, stage, query_pool, query);
}

auto CommandBuffer::image_memory_barrier(VkImage image, VkImageAspectFlags aspect_mask,
                                          VkImageLayout old_layout, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                          VkImageLayout new_layout, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                                          uint32_t src_queue_family, uint32_t dst_queue_family) -> void {
    VkImageMemoryBarrier2 barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = src_stage;
    barrier.srcAccessMask = src_access;
    barrier.dstStageMask = dst_stage;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = src_queue_family;
    barrier.dstQueueFamilyIndex = dst_queue_family;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspect_mask;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    
    VkDependencyInfo dependency_info = {};
    dependency_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency_info.imageMemoryBarrierCount = 1;
    dependency_info.pImageMemoryBarriers = &barrier;
    
    vkCmdPipelineBarrier2(_cmd, &dependency_info);
}

auto CommandBuffer::set_viewport(float x, float y, float width, float height, float min_depth, float max_depth) -> void {
    VkViewport viewport = {};
    viewport.x = x;
    viewport.y = y;
    viewport.width = width;
    viewport.height = height;
    viewport.minDepth = min_depth;
    viewport.maxDepth = max_depth;
    vkCmdSetViewportWithCount(_cmd, 1, &viewport);
}

auto CommandBuffer::set_scissor(int32_t x, int32_t y, uint32_t width, uint32_t height) -> void {
    VkRect2D scissor = {};
    scissor.offset = {x, y};
    scissor.extent = {width, height};
    vkCmdSetScissorWithCount(_cmd, 1, &scissor);
}

auto CommandBuffer::set_rasterizer_discard_enable(bool enable) -> void {
    vkCmdSetRasterizerDiscardEnable(_cmd, enable ? VK_TRUE : VK_FALSE);
}

auto CommandBuffer::set_polygon_mode(VkPolygonMode polygon_mode) -> void {
    vkCmdSetPolygonModeEXT(_cmd, polygon_mode);
}

auto CommandBuffer::set_cull_mode(VkCullModeFlags cull_mode) -> void {
    vkCmdSetCullMode(_cmd, cull_mode);
}

auto CommandBuffer::set_front_face(VkFrontFace front_face) -> void {
    vkCmdSetFrontFace(_cmd, front_face);
}

auto CommandBuffer::set_depth_bias_enable(bool enable) -> void {
    vkCmdSetDepthBiasEnable(_cmd, enable ? VK_TRUE : VK_FALSE);
}

auto CommandBuffer::set_primitive_topology(VkPrimitiveTopology topology) -> void {
    vkCmdSetPrimitiveTopology(_cmd, topology);
}

auto CommandBuffer::set_primitive_restart_enable(bool enable) -> void {
    vkCmdSetPrimitiveRestartEnable(_cmd, enable ? VK_TRUE : VK_FALSE);
}

auto CommandBuffer::set_depth_test_enable(bool enable) -> void {
    vkCmdSetDepthTestEnable(_cmd, enable ? VK_TRUE : VK_FALSE);
}

auto CommandBuffer::set_depth_write_enable(bool enable) -> void {
    vkCmdSetDepthWriteEnable(_cmd, enable ? VK_TRUE : VK_FALSE);
}

auto CommandBuffer::set_depth_bounds_test_enable(bool enable) -> void {
    vkCmdSetDepthBoundsTestEnable(_cmd, enable ? VK_TRUE : VK_FALSE);
}

auto CommandBuffer::set_stencil_test_enable(bool enable) -> void {
    vkCmdSetStencilTestEnable(_cmd, enable ? VK_TRUE : VK_FALSE);
}

auto CommandBuffer::set_rasterization_samples(VkSampleCountFlagBits samples) -> void {
    vkCmdSetRasterizationSamplesEXT(_cmd, samples);
}

auto CommandBuffer::set_sample_mask(VkSampleCountFlagBits samples, uint32_t mask) -> void {
    VkSampleMask sample_mask = mask;
    vkCmdSetSampleMaskEXT(_cmd, samples, &sample_mask);
}

auto CommandBuffer::set_alpha_to_coverage_enable(bool enable) -> void {
    vkCmdSetAlphaToCoverageEnableEXT(_cmd, enable ? VK_TRUE : VK_FALSE);
}

auto CommandBuffer::set_color_blend_enable(uint32_t attachment, bool enable) -> void {
    VkBool32 enables[] = {enable ? VK_TRUE : VK_FALSE};
    vkCmdSetColorBlendEnableEXT(_cmd, attachment, 1, enables);
}

auto CommandBuffer::set_color_write_mask(uint32_t attachment, VkColorComponentFlags mask) -> void {
    VkColorComponentFlags masks[] = {mask};
    vkCmdSetColorWriteMaskEXT(_cmd, attachment, 1, masks);
}

auto CommandBuffer::set_vertex_input_empty() -> void {
    vkCmdSetVertexInputEXT(_cmd, 0, nullptr, 0, nullptr);
}

auto CommandBuffer::bind_descriptor_sets(VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                                          uint32_t first_set, uint32_t set_count, VkDescriptorSet const* sets) -> void {
    vkCmdBindDescriptorSets(_cmd, bind_point, layout, first_set, set_count, sets, 0, nullptr);
}

auto CommandBuffer::push_constants(VkPipelineLayout layout, VkShaderStageFlags stage_flags,
                                    uint32_t offset, uint32_t size, void const* data) -> void {
    vkCmdPushConstants(_cmd, layout, stage_flags, offset, size, data);
}

auto CommandBuffer::draw(uint32_t vertex_count, uint32_t instance_count, 
                          uint32_t first_vertex, uint32_t first_instance) -> void {
    vkCmdDraw(_cmd, vertex_count, instance_count, first_vertex, first_instance);
}
