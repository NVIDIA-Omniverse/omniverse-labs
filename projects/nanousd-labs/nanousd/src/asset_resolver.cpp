// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/asset_resolver.h"
#include "nanousd/usdz_package.h"
#include "uri.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <optional>
#include <string_view>

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

bool IsWindowsDriveAbsolutePath(std::string_view path) {
    return path.size() >= 3 && detail::HasWindowsDrivePrefix(path) &&
           (path[2] == '/' || path[2] == '\\');
}

bool IsWindowsUncPath(std::string_view path) {
    return path.size() >= 2 &&
           ((path[0] == '\\' && path[1] == '\\') ||
            (path[0] == '/' && path[1] == '/'));
}

bool IsWindowsRootRelativePath(std::string_view path) {
    return !path.empty() && path[0] == '\\' && !IsWindowsUncPath(path);
}

bool ContainsBackslash(std::string_view path) {
    return path.find('\\') != std::string_view::npos;
}

bool IsAnchoredRelativeIdentifier(std::string_view path) {
    // AOUSD Core Spec 1.0.1 §9.4.1 anchors ./ and ../ to the authored
    // document; §9.4.2 leaves non-anchored relatives to application search
    // paths. Backslash forms mirror this for non-normative Windows paths.
    return detail::StartsWith(path, "./") || detail::StartsWith(path, "../") ||
           detail::StartsWith(path, ".\\") || detail::StartsWith(path, "..\\");
}

bool IsWindowsPathLike(std::string_view path) {
    return detail::HasWindowsDrivePrefix(path) || IsWindowsUncPath(path) ||
           ContainsBackslash(path);
}

std::string ToForwardSlashes(std::string_view path) {
    std::string out(path);
    std::replace(out.begin(), out.end(), '\\', '/');
    return out;
}

std::string WindowsDrivePrefix(std::string_view path) {
    return detail::HasWindowsDrivePrefix(path) ? std::string(path.substr(0, 2)) : std::string();
}

std::string WindowsParentPath(std::string_view path) {
    std::string normalized = ToForwardSlashes(path);
    size_t slash = normalized.find_last_of('/');
    if (slash == std::string::npos) return {};
    return normalized.substr(0, slash);
}

std::string ResolveWindowsRelativePath(std::string_view anchorLayerPath,
                                       std::string_view assetPath) {
    std::string parent = WindowsParentPath(anchorLayerPath);
    if (parent.empty()) return {};

    std::string asset = ToForwardSlashes(assetPath);
    std::string joined = parent;
    if (!joined.empty() && joined.back() != '/') joined.push_back('/');
    joined.append(asset);
    return detail::RemoveUriDotSegments(joined);
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

std::optional<std::filesystem::path> TryLocalFilesystemLayerPath(
        const std::string& anchorLayerPath) {
    namespace fs = std::filesystem;
    if (anchorLayerPath.empty()) return std::nullopt;
    if (IsWindowsPathLike(anchorLayerPath)) {
        return fs::path(ToForwardSlashes(anchorLayerPath));
    }

    detail::UriParseResult anchorUri = detail::ParseUriReference(anchorLayerPath);
    if (anchorUri.success && anchorUri.uri.scheme) {
        if (auto filePath = TryLocalFileUriPath(anchorUri.uri)) {
            return fs::path(*filePath);
        }
        return std::nullopt;
    }
    if (!anchorUri.success && detail::HasUriSchemePrefix(anchorLayerPath)) {
        return std::nullopt;
    }
    return fs::path(anchorLayerPath);
}

std::string ResolvedUriLocation(const detail::UriReference& uri) {
    if (auto filePath = TryLocalFileUriPath(uri)) return *filePath;
    return detail::FormatUriReference(uri);
}

} // namespace

std::string DefaultResolve(const std::string& anchorLayerPath,
                           const std::string& assetPath) {
    namespace fs = std::filesystem;
    if (assetPath.empty()) return {};

    // Resolve explicit packaged-resource identifiers by resolving the
    // package file itself and preserving the requested inner resource.
    std::string assetPackage;
    std::string assetEntry;
    if (SplitPackageIdentifier(assetPath, &assetPackage, &assetEntry)) {
        std::string resolvedPackage = DefaultResolve(anchorLayerPath, assetPackage);
        if (resolvedPackage.empty()) resolvedPackage = assetPackage;
        return MakePackageIdentifier(resolvedPackage, assetEntry);
    }

    if (IsWindowsDriveAbsolutePath(assetPath) || IsWindowsUncPath(assetPath)) {
        return assetPath;
    }
    if (detail::HasWindowsDrivePrefix(assetPath)) {
        return assetPath;
    }
    if (IsWindowsRootRelativePath(assetPath)) {
        std::string drive = WindowsDrivePrefix(anchorLayerPath);
        if (!drive.empty()) return drive + ToForwardSlashes(assetPath);
        return assetPath;
    }

    // POSIX filesystem paths are absolute in the default resolver. They are
    // not URI path-absolute references against URI-shaped layer anchors.
    if (assetPath.front() == '/') return assetPath;

    const bool anchoredRelative = IsAnchoredRelativeIdentifier(assetPath);
    std::string uriAssetPath = anchoredRelative ? ToForwardSlashes(assetPath) : assetPath;

    detail::UriParseResult assetUri = detail::ParseUriReference(uriAssetPath);
    if (assetUri.success && assetUri.uri.scheme) {
        return ResolvedUriLocation(assetUri.uri);
    }
    if (!assetUri.success && detail::HasUriSchemePrefix(assetPath)) {
        return assetPath;
    }

    // When a layer came from a USDZ package, package semantics resolve
    // relative identifiers to resources inside the package. The package path
    // itself has already been resolved by the normal URI/filesystem layer.
    std::string anchorPackage;
    std::string anchorEntry;
    bool anchorIsPackage = SplitPackageIdentifier(anchorLayerPath,
                                                  &anchorPackage,
                                                  &anchorEntry);
    if (!anchorIsPackage && HasUsdzExtension(anchorLayerPath)) {
        anchorPackage = anchorLayerPath;
        anchorEntry = GetUsdzRootLayerPath(anchorLayerPath);
        anchorIsPackage = !anchorEntry.empty();
    }
    if (anchorIsPackage) {
        return MakePackageIdentifier(anchorPackage,
                                     JoinPackageEntryPath(anchorEntry, assetPath));
    }

    detail::UriParseResult anchorUri = detail::ParseUriReference(anchorLayerPath);
    if (anchoredRelative && assetUri.success && anchorUri.success && anchorUri.uri.scheme) {
        detail::UriParseResult resolved =
            detail::ResolveUriReference(anchorLayerPath, uriAssetPath);
        if (resolved.success) return ResolvedUriLocation(resolved.uri);
    }

    if (anchoredRelative && IsWindowsPathLike(anchorLayerPath)) {
        std::string resolved = ResolveWindowsRelativePath(anchorLayerPath, assetPath);
        if (!resolved.empty()) return resolved;
    }

    fs::path asset(assetPath);
    if (asset.is_absolute()) return assetPath;
    if (detail::HasUriSchemePrefix(assetPath)) return assetPath;

    if (!anchoredRelative) {
        if (auto anchorPath = TryLocalFilesystemLayerPath(anchorLayerPath)) {
            fs::path dir = anchorPath->parent_path();
            if (!dir.empty()) {
                fs::path resolved = dir / asset;
                std::error_code ec;
                if (fs::exists(resolved, ec)) {
                    auto canonical = fs::weakly_canonical(resolved, ec);
                    return ec ? resolved.generic_string() : canonical.generic_string();
                }
            }
        }
        return assetPath;
    }

    fs::path anchor(anchorLayerPath);
    fs::path dir = anchor.parent_path();
    fs::path resolved = (dir / asset).lexically_normal();

    // generic_string() emits forward slashes on every host OS; .string()
    // would give backslashes on Windows, which breaks USDA round-trips
    // and forward-slash-literal substring assertions downstream.
    return resolved.generic_string();
}

} // namespace nanousd
