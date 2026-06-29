// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "layer.h"
#include "resource.h"

#include <string>

namespace nanousd {

// Serialize a Layer to USDA text format.
// Returns a valid USDA string starting with "#usda 1.0\n", or empty on error.
NANOUSD_CORE_API std::string WriteUsda(const Layer& layer);

// Serialize a Layer to a USDA text file.
// Returns true on success, false if the file could not be written.
NANOUSD_CORE_API bool WriteUsdaFile(const Layer& layer, const ResolvedLocation& location);
NANOUSD_CORE_API bool WriteUsdaFile(const Layer& layer, const std::string& path);

} // namespace nanousd
