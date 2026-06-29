// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/resource.h"
#include "nanousd/usdz_package.h"
#include "uri.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>

namespace nanousd {
namespace {

std::string ToLowerAscii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool IsLocalhostAuthority(const std::optional<std::string>& authority) {
    if (!authority || authority->empty()) return true;
    return ToLowerAscii(*authority) == "localhost";
}

bool IsWindowsUncPath(std::string_view path) {
    return path.size() >= 2 &&
           ((path[0] == '\\' && path[1] == '\\') ||
            (path[0] == '/' && path[1] == '/'));
}

bool IsWindowsPathLike(std::string_view path) {
    return detail::HasWindowsDrivePrefix(path) ||
           IsWindowsUncPath(path) ||
           path.find('\\') != std::string_view::npos;
}

int FromHex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

std::string PercentDecode(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '%' && i + 2 < text.size()) {
            int hi = FromHex(text[i + 1]);
            int lo = FromHex(text[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        out.push_back(text[i]);
    }
    return out;
}

std::optional<std::string> TryLocalFileUriPath(const detail::UriReference& uri) {
    if (!uri.scheme || ToLowerAscii(*uri.scheme) != "file") return std::nullopt;
    if (uri.query || uri.fragment) return std::nullopt;
    if (!IsLocalhostAuthority(uri.authority)) return std::nullopt;

    std::string path = PercentDecode(uri.path);
    if (path.size() >= 3 && path[0] == '/' &&
            std::isalpha(static_cast<unsigned char>(path[1])) &&
            path[2] == ':') {
        path.erase(path.begin());
    }
    return path;
}

std::optional<std::string> TryLenientLocalFileUriPath(std::string_view text) {
    constexpr std::string_view kFilePrefix = "file:";
    if (text.size() < kFilePrefix.size()) return std::nullopt;
    for (size_t i = 0; i < kFilePrefix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(text[i])) != kFilePrefix[i])
            return std::nullopt;
    }

    std::string_view rest = text.substr(kFilePrefix.size());
    if (rest.find('?') != std::string_view::npos ||
        rest.find('#') != std::string_view::npos) {
        return std::nullopt;
    }

    if (detail::StartsWith(rest, "///")) {
        rest.remove_prefix(3);
    } else if (detail::StartsWith(rest, "//localhost/")) {
        rest.remove_prefix(12);
    } else if (detail::StartsWith(rest, "/")) {
        rest.remove_prefix(1);
    } else {
        return std::nullopt;
    }

    std::string path = PercentDecode(rest);
    if (!detail::HasWindowsDrivePrefix(path) &&
        !IsWindowsUncPath(path) &&
        path.find('\\') == std::string::npos) {
        return std::nullopt;
    }
    return path;
}

std::string UnsupportedReadError(const ResolvedLocation& location) {
    auto parsed = detail::ParseUriReference(location.DisplayString());
    if (!parsed.success && detail::HasUriSchemePrefix(location.DisplayString()))
        return "Invalid resource URI: " + parsed.error;
    if (parsed.success && parsed.uri.scheme) {
        std::string scheme = ToLowerAscii(*parsed.uri.scheme);
        if (scheme == "file")
            return "Unsupported file resource location: " + location.DisplayString();
        return "Unsupported resource scheme: " + scheme;
    }
    return "Unsupported resource location: " + location.DisplayString();
}

std::string UnsupportedWriteError(const ResolvedLocation& location) {
    auto parsed = detail::ParseUriReference(location.DisplayString());
    if (!parsed.success && detail::HasUriSchemePrefix(location.DisplayString()))
        return "Invalid resource URI for write: " + parsed.error;
    if (parsed.success && parsed.uri.scheme) {
        std::string scheme = ToLowerAscii(*parsed.uri.scheme);
        if (scheme == "file")
            return "Unsupported file resource location for write: " + location.DisplayString();
        return "Unsupported resource scheme for write: " + scheme;
    }
    return "Unsupported resource location for write: " + location.DisplayString();
}

} // anonymous namespace

ResolvedLocation ResolvedLocation::LocalFile(std::string path,
                                             std::string display) {
    ResolvedLocation location;
    location.type_ = Type::LocalFile;
    location.filePath_ = std::move(path);
    location.display_ = display.empty() ? location.filePath_ : std::move(display);
    return location;
}

ResolvedLocation ResolvedLocation::Uri(std::string uri) {
    ResolvedLocation location;
    location.type_ = Type::Uri;
    location.uri_ = std::move(uri);
    location.display_ = location.uri_;
    return location;
}

ResolvedLocation ResolvedLocation::Package(ResolvedLocation package,
                                           std::string entryPath) {
    ResolvedLocation location;
    location.type_ = Type::Package;
    location.package_ = std::make_shared<ResolvedLocation>(std::move(package));
    location.entryPath_ = NormalizePackageEntryPath(entryPath);
    location.display_ = MakePackageIdentifier(location.package_->DisplayString(),
                                              location.entryPath_);
    return location;
}

ResolvedLocation ResolvedLocation::FromResolvedString(std::string resolved) {
    std::string packagePath;
    std::string entryPath;
    if (SplitPackageIdentifier(resolved, &packagePath, &entryPath)) {
        return Package(FromResolvedString(std::move(packagePath)),
                       std::move(entryPath));
    }

    if (IsWindowsPathLike(resolved) &&
        !detail::HasUriSchemePrefix(resolved)) {
        return LocalFile(std::move(resolved));
    }

    auto parsed = detail::ParseUriReference(resolved);
    if (parsed.success && parsed.uri.scheme) {
        if (auto filePath = TryLocalFileUriPath(parsed.uri)) {
            return LocalFile(std::move(*filePath), std::move(resolved));
        }
        return Uri(std::move(resolved));
    }
    if (!parsed.success) {
        if (auto filePath = TryLenientLocalFileUriPath(resolved)) {
            return LocalFile(std::move(*filePath), std::move(resolved));
        }
    }
    if (!parsed.success && detail::HasUriSchemePrefix(resolved)) {
        return Uri(std::move(resolved));
    }

    return LocalFile(std::move(resolved));
}

const ResolvedLocation& ResolvedLocation::PackageLocation() const {
    static const ResolvedLocation empty;
    return package_ ? *package_ : empty;
}

bool IsLocalFileResource(const ResolvedLocation& location,
                         std::string* filePath) {
    if (!location.IsLocalFile()) return false;
    if (filePath) *filePath = location.LocalFilePath();
    return true;
}

bool IsLocalFileResource(const std::string& resolvedLocation,
                         std::string* filePath) {
    return IsLocalFileResource(ResolvedLocation::FromResolvedString(resolvedLocation),
                               filePath);
}

ResourceReadResult ReadResource(const ResolvedLocation& location) {
    ResourceReadResult result;
    result.resolvedLocation = location.DisplayString();

    if (location.IsPackage()) {
        auto package = ReadUsdzFile(location.DisplayString());
        if (!package.success) {
            result.error = std::move(package.error);
            return result;
        }
        result.bytes = std::move(package.data);
        result.success = true;
        return result;
    }
    if (!location.IsLocalFile()) {
        result.error = UnsupportedReadError(location);
        return result;
    }

    result.fileBacked = true;
    result.filePath = location.LocalFilePath();

    std::ifstream file(result.filePath, std::ios::binary);
    if (!file) {
        result.error = "Failed to open resource: " + location.DisplayString();
        return result;
    }

    file.seekg(0, std::ios::end);
    std::streamoff end = file.tellg();
    if (end < 0) {
        file.clear();
        file.seekg(0, std::ios::beg);
        result.bytes.assign(std::istreambuf_iterator<char>(file),
                            std::istreambuf_iterator<char>());
        if (file.bad()) {
            result.error = "Failed to read resource: " + location.DisplayString();
            result.bytes.clear();
            return result;
        }
        result.success = true;
        return result;
    }
    if (static_cast<uint64_t>(end) >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        result.error = "Resource is too large to read: " + location.DisplayString();
        return result;
    }

    result.bytes.resize(static_cast<size_t>(end));
    file.seekg(0, std::ios::beg);
    if (!result.bytes.empty()) {
        file.read(reinterpret_cast<char*>(result.bytes.data()),
                  static_cast<std::streamsize>(result.bytes.size()));
        if (!file) {
            result.error = "Failed to read resource: " + location.DisplayString();
            result.bytes.clear();
            return result;
        }
    }

    result.success = true;
    return result;
}

ResourceReadResult ReadResource(const std::string& resolvedLocation) {
    return ReadResource(ResolvedLocation::FromResolvedString(resolvedLocation));
}

ResourceWriteResult WriteResource(const ResolvedLocation& location,
                                  const uint8_t* data,
                                  size_t size) {
    ResourceWriteResult result;
    result.resolvedLocation = location.DisplayString();

    if (location.IsPackage()) {
        result.error = "Writing packaged resources is not supported: " +
                       location.DisplayString();
        return result;
    }
    if (!location.IsLocalFile()) {
        result.error = UnsupportedWriteError(location);
        return result;
    }
    if (size > 0 && data == nullptr) {
        result.error = "Cannot write resource from null data: " +
                       location.DisplayString();
        return result;
    }

    result.fileBacked = true;
    result.filePath = location.LocalFilePath();

    std::ofstream file(result.filePath, std::ios::binary | std::ios::trunc);
    if (!file) {
        result.error = "Failed to open resource for write: " +
                       location.DisplayString();
        return result;
    }
    if (size > 0) {
        file.write(reinterpret_cast<const char*>(data),
                   static_cast<std::streamsize>(size));
        if (!file) {
            result.error = "Failed to write resource: " +
                           location.DisplayString();
            return result;
        }
    }
    file.close();
    if (!file) {
        result.error = "Failed to close resource after write: " +
                       location.DisplayString();
        return result;
    }

    result.success = true;
    return result;
}

ResourceWriteResult WriteResource(const std::string& resolvedLocation,
                                  const uint8_t* data,
                                  size_t size) {
    return WriteResource(ResolvedLocation::FromResolvedString(resolvedLocation),
                         data,
                         size);
}

} // namespace nanousd
