// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "validation_tracker.hpp"
#include <gtest/gtest.h>

auto ValidationTracker::instance() -> ValidationTracker& {
    static ValidationTracker tracker;
    return tracker;
}

auto ValidationTracker::callback() -> DebugCallback {
    return [this](DebugSeverity severity, std::string_view message) {
        // Only track warnings and errors - ignore INFO and VERBOSE
        if (severity != DebugSeverity::Warning && severity != DebugSeverity::Error) {
            return;
        }
        
        std::lock_guard<std::mutex> lock(_mutex);
        _has_warnings = true;
        
        char const* severity_str = (severity == DebugSeverity::Error) ? "ERROR" : "WARNING";
        _messages.push_back(std::string("[") + severity_str + "] " + std::string(message));
    };
}

auto ValidationTracker::has_warnings_or_errors() const -> bool {
    std::lock_guard<std::mutex> lock(_mutex);
    return _has_warnings;
}

auto ValidationTracker::messages() const -> std::vector<std::string> const& {
    return _messages;
}

auto ValidationTracker::clear() -> void {
    std::lock_guard<std::mutex> lock(_mutex);
    _messages.clear();
    _has_warnings = false;
}

auto ValidationTracker::check_and_clear() -> void {
    std::lock_guard<std::mutex> lock(_mutex);
    
    if (_has_warnings) {
        std::string all_messages;
        for (std::string const& msg : _messages) {
            all_messages += msg + "\n";
        }
        
        _messages.clear();
        _has_warnings = false;
        
        GTEST_FAIL() << "Vulkan validation layer produced warnings/errors:\n" << all_messages;
    }
    
    _messages.clear();
    _has_warnings = false;
}

