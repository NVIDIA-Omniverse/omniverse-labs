// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "layer.h"
#include "resource.h"

#include <string>

namespace nanousd {

// Unified parse result that covers both USDA and USDC formats.
struct UsdParseResult {
    bool success = false;
    std::string error;
    int line = 0;     // USDA only (0 for USDC)
    int column = 0;   // USDA only (0 for USDC)
    Layer layer;
};

// Parse a USD layer resource (USDA, USDC, or USDZ) with automatic format detection.
// Checks the first 8 bytes for the "PXR-USDC" magic; if found, parses as
// USDC binary. Otherwise, parses as USDA text.
NANOUSD_CORE_API UsdParseResult ParseUsdFile(const ResolvedLocation& location);
NANOUSD_CORE_API UsdParseResult ParseUsdFile(const std::string& filePath);

} // namespace nanousd
