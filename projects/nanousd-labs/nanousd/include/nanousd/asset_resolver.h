// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"

#include <functional>
#include <string>

namespace nanousd {

// Callback to resolve an asset identifier using a layer anchor as context.
// Returns the resolved file path, or empty string on failure.
using AssetResolver = std::function<std::string(const std::string& anchorLayerPath,
                                                 const std::string& assetPath)>;

// Default asset resolver: resolves explicitly anchored relative asset paths
// (./ and ../) against anchorLayerPath. Non-anchored relative identifiers are
// returned unchanged; applications that need search-path lookup should provide
// a custom resolver.
NANOUSD_CORE_API std::string DefaultResolve(const std::string& anchorLayerPath,
                                            const std::string& assetPath);

} // namespace nanousd
