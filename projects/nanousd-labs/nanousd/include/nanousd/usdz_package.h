// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "layer.h"
#include "resource.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace nanousd {

struct UsdzFileEntry {
    std::string path;
    std::vector<uint8_t> data;
};

struct UsdzReadResult {
    bool success = false;
    std::string error;
    std::string packagePath;
    std::string entryPath;
    std::vector<uint8_t> data;
};

struct UsdzWriteResult {
    bool success = false;
    std::string error;
    std::vector<uint8_t> data;
};

NANOUSD_CORE_API bool HasUsdzExtension(std::string_view path);
NANOUSD_CORE_API bool IsPackageIdentifier(std::string_view identifier);
NANOUSD_CORE_API bool SplitPackageIdentifier(std::string_view identifier,
                                             std::string* packagePath,
                                             std::string* entryPath);

NANOUSD_CORE_API std::string NormalizePackageEntryPath(std::string_view path);
NANOUSD_CORE_API std::string JoinPackageEntryPath(std::string_view anchorEntry,
                                                  std::string_view relativePath);
NANOUSD_CORE_API std::string MakePackageIdentifier(const std::string& packagePath,
                                                   const std::string& entryPath);

NANOUSD_CORE_API std::string GetUsdzRootLayerPath(const std::string& packagePath,
                                                  std::string* error = nullptr);

NANOUSD_CORE_API UsdzReadResult ReadUsdzFile(const std::string& packagePathOrIdentifier);

NANOUSD_CORE_API UsdzWriteResult WriteUsdz(const std::vector<UsdzFileEntry>& entries);
NANOUSD_CORE_API UsdzWriteResult WriteUsdz(const Layer& rootLayer,
                                           std::string_view rootEntryPath = "root.usdc");
NANOUSD_CORE_API bool WriteUsdzFile(const std::vector<UsdzFileEntry>& entries,
                                    const ResolvedLocation& location,
                                    std::string* error = nullptr);
NANOUSD_CORE_API bool WriteUsdzFile(const std::vector<UsdzFileEntry>& entries,
                                    const std::string& path,
                                    std::string* error = nullptr);
NANOUSD_CORE_API bool WriteUsdzFile(const Layer& rootLayer,
                                    const ResolvedLocation& location,
                                    std::string_view rootEntryPath = "root.usdc",
                                    std::string* error = nullptr);
NANOUSD_CORE_API bool WriteUsdzFile(const Layer& rootLayer,
                                    const std::string& path,
                                    std::string_view rootEntryPath = "root.usdc",
                                    std::string* error = nullptr);

} // namespace nanousd
