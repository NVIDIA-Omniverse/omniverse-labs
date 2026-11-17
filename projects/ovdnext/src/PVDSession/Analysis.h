/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace ovdnext {
struct MessageInfo {
    uint64_t frameId;
    std::string typeName, message, file;
    uint32_t line, type, handle;
};
struct FrameTiming {
    bool supported = false;
    double innerMs = 0, totalMs = 0;
};
struct ScalarHistory {
    std::string propertyName, type;
    std::vector<uint32_t> frames;
    // Graph ordinates use doubles. Int64/Uint64 above 2**53 can lose precision;
    // lossless scalar inspection/export should use Session.read instead.
    std::vector<double> values;
    std::vector<uint8_t> valid;
};
}
