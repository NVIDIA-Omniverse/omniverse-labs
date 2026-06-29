// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nanousd {

// Typed resource location produced after asset identifier resolution.
// This keeps authored identifiers, resolved filesystem/URI locations, and
// package-entry locations distinct at resource I/O boundaries.
class NANOUSD_CORE_API ResolvedLocation {
public:
    enum class Type {
        LocalFile,
        Uri,
        Package,
    };

    ResolvedLocation() = default;

    static ResolvedLocation LocalFile(std::string path,
                                      std::string display = {});
    static ResolvedLocation Uri(std::string uri);
    static ResolvedLocation Package(ResolvedLocation package,
                                    std::string entryPath);
    static ResolvedLocation FromResolvedString(std::string resolved);

    Type type() const { return type_; }
    const std::string& DisplayString() const { return display_; }

    bool IsLocalFile() const { return type_ == Type::LocalFile; }
    const std::string& LocalFilePath() const { return filePath_; }

    bool IsUri() const { return type_ == Type::Uri; }
    const std::string& UriString() const { return uri_; }

    bool IsPackage() const { return type_ == Type::Package; }
    const ResolvedLocation& PackageLocation() const;
    const std::string& PackageEntryPath() const { return entryPath_; }

private:
    Type type_ = Type::LocalFile;
    std::string display_;
    std::string filePath_;
    std::string uri_;
    std::shared_ptr<ResolvedLocation> package_;
    std::string entryPath_;
};

// Result of opening and reading a resolved resource location.
//
// Asset resolution returns resolver output. ResolvedLocation typing is the next
// layer: resource reading maps that typed location to bytes without changing
// identifier semantics. The default implementation supports local filesystem
// resources, including local file:// URIs, and reports unsupported schemes
// explicitly.
struct ResourceReadResult {
    bool success = false;
    std::string error;
    std::string resolvedLocation;
    std::vector<uint8_t> bytes;
    bool fileBacked = false;
    std::string filePath;
};

struct ResourceWriteResult {
    bool success = false;
    std::string error;
    std::string resolvedLocation;
    bool fileBacked = false;
    std::string filePath;
};

// Return true if the resolved location names a local filesystem resource. When
// filePath is non-null it receives the decoded local path to open.
NANOUSD_CORE_API bool IsLocalFileResource(const ResolvedLocation& location,
                                          std::string* filePath = nullptr);
NANOUSD_CORE_API bool IsLocalFileResource(const std::string& resolvedLocation,
                                          std::string* filePath = nullptr);

// Read a resolved resource location into memory. This is intentionally small:
// plain paths and local file:// URIs use the filesystem; other schemes are
// rejected with a clear diagnostic.
NANOUSD_CORE_API ResourceReadResult ReadResource(const ResolvedLocation& location);
NANOUSD_CORE_API ResourceReadResult ReadResource(const std::string& resolvedLocation);

// Write a complete resource. The default implementation supports local
// filesystem resources, including local file:// URIs. Package-entry writes and
// non-file URI writes are reported as unsupported.
NANOUSD_CORE_API ResourceWriteResult WriteResource(const ResolvedLocation& location,
                                                   const uint8_t* data,
                                                   size_t size);
NANOUSD_CORE_API ResourceWriteResult WriteResource(const std::string& resolvedLocation,
                                                   const uint8_t* data,
                                                   size_t size);

} // namespace nanousd
