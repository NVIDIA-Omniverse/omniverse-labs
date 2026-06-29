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

#include "vk/vulkan_context.hpp"
#include <vector>
#include <string>
#include <mutex>

// Singleton class to track Vulkan validation layer messages during tests
class ValidationTracker {
public:
    static auto instance() -> ValidationTracker&;
    
    // Get a debug callback that records messages to this tracker
    auto callback() -> DebugCallback;
    
    // Check if any warnings or errors have been recorded
    auto has_warnings_or_errors() const -> bool;
    
    // Get all recorded messages
    auto messages() const -> std::vector<std::string> const&;
    
    // Clear all recorded messages
    auto clear() -> void;
    
    // Check for warnings/errors and call GTEST_FAIL if any, then clear
    auto check_and_clear() -> void;
    
private:
    ValidationTracker() = default;
    
    mutable std::mutex _mutex;
    std::vector<std::string> _messages;
    bool _has_warnings = false;
};

