// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "layer.h"
#include "resource.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nanousd {

// Write a Layer to a USDC binary crate file (spec 16.3).
// Returns true on success, false on error.
NANOUSD_CORE_API bool WriteUsdcFile(const Layer& layer, const ResolvedLocation& location);
NANOUSD_CORE_API bool WriteUsdcFile(const Layer& layer, const std::string& path);

// Serialise a Layer to a USDC binary crate in memory.
// Returns an empty vector on error.
NANOUSD_CORE_API std::vector<uint8_t> WriteUsdc(const Layer& layer);

} // namespace nanousd
