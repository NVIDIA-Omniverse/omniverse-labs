// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "layer.h"
#include "resource.h"

#include <optional>
#include <string>
#include <string_view>

namespace nanousd {

// Parse result from USDA text format parsing.
// On success, the layer field contains the parsed data.
struct UsdaParseResult {
    bool success = false;
    std::string error;
    int line = 0;
    int column = 0;
    Layer layer;
};

// Parse a USDA (text format) string.
// Implements the grammar from spec Section 16.2.
NANOUSD_CORE_API UsdaParseResult ParseUsda(std::string_view text);

// Parse a USDA layer from a resolved resource location.
NANOUSD_CORE_API UsdaParseResult ParseUsdaFile(const ResolvedLocation& location);
NANOUSD_CORE_API UsdaParseResult ParseUsdaFile(const std::string& filePath);

} // namespace nanousd
