// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "layer.h"
#include "resource.h"

#include <cstdint>
#include <string>

namespace nanousd {

// Result of parsing a USDC (Crate binary) file.
struct UsdcParseResult {
    bool success = false;
    std::string error;
    Layer layer;
};

// Parse a USDC file from a memory buffer.
NANOUSD_CORE_API UsdcParseResult ParseUsdc(const uint8_t* data, size_t size);

// Parse a USDC layer from a resolved resource location.
NANOUSD_CORE_API UsdcParseResult ParseUsdcFile(const ResolvedLocation& location);
NANOUSD_CORE_API UsdcParseResult ParseUsdcFile(const std::string& filePath);

// Check if a buffer starts with the USDC magic bytes ("PXR-USDC").
NANOUSD_CORE_API bool IsUsdcFormat(const uint8_t* data, size_t size);

} // namespace nanousd
